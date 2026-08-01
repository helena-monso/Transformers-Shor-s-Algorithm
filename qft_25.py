#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25-Qubit QFT on a 1D Cluster State

This script benchmarks the Transformer's ability to track the dynamic redistribution 
of entanglement by applying a 25-qubit QFT to an initial 1D cluster state.
"""

import os
# MEMORY OPTIMIZATIONS
# Pre-allocation is disabled to prevent JAX from hoarding GPU memory, 
# allowing for exact math operations to run alongside the neural network.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import functools
import itertools
import numpy as np
import scipy.linalg as sla
from tqdm.auto import tqdm
import json
import orbax.checkpoint as ocp
import pathlib


# REMOTE LOGGING SETUP 
num_qubits = 25

log_file = open(f"qft_{num_qubits}_cluster_output.log", "w", encoding="utf-8")

def log_print(*args, **kwargs):
    print(*args, **kwargs) 
    print(*args, file=log_file, **kwargs)
    log_file.flush() 

# Initialize tracking files for loss and fidelity
with open(f"qft_{num_qubits}_loss.txt", "w") as f:
    f.write("step_name,step,loss\n")
with open(f"qft_{num_qubits}_fidelity_tracker.txt", "w") as f:
    f.write("gate_index,gate_name,transition_fidelity\n")


# 1. CPU MATRIX GENERATION & POVM DEFINITIONS
# We use NumPy/SciPy on the CPU in 64-bit precision to compute exact theoretical 
# transition matrices. This prevents floating-point drift over deep circuits, 
# separating the exact physical targets from the JAX 32-bit network operations.

I_np = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X_np = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_np = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_np = np.array([[1, 0], [0, -1]], dtype=np.complex128)

c_np = 1.0 / np.sqrt(3)
Pi_1_np = 0.5 * (I_np + c_np * (X_np - Y_np + Z_np))
Pi_2_np = 0.5 * (I_np + c_np * (X_np + Y_np - Z_np))
Pi_3_np = 0.5 * (I_np + c_np * (-X_np + Y_np + Z_np))
Pi_4_np = 0.5 * (I_np + c_np * (-X_np - Y_np - Z_np))

SIC_PROJECTORS_NP = np.array([Pi_1_np, Pi_2_np, Pi_3_np, Pi_4_np])
SIC_PROJECTORS = jnp.array(SIC_PROJECTORS_NP, dtype=jnp.complex128)

IC_PROJECTORS_2Q_list_np = [np.kron(P1, P2) for P1 in SIC_PROJECTORS_NP for P2 in SIC_PROJECTORS_NP]
IC_PROJECTORS_2Q_NP = np.array(IC_PROJECTORS_2Q_list_np)

def get_1q_S_matrix_np(U, projectors=SIC_PROJECTORS_NP):
    """Generates the pseudo-bistochastic S matrix for a 1-qubit gate."""
    num_outcomes = projectors.shape[0]
    d = int(np.sqrt(num_outcomes))
    S_matrix = np.zeros((num_outcomes, num_outcomes), dtype=np.float64)
    U_dag = np.conj(U.T)
    for i in range(num_outcomes):
        for j in range(num_outcomes):
            evolved_Pi_j = U @ projectors[j] @ U_dag
            trace_val = np.trace(evolved_Pi_j @ projectors[i])
            s_ij = (1.0 / d) * np.real(trace_val)
            S_matrix[i, j] = (d + 1) * s_ij - (1.0 / d)
    return jnp.array(S_matrix, dtype=jnp.float64)

def get_2q_O_matrix_np(U_2q, projectors_2q=IC_PROJECTORS_2Q_NP):
    """Generates the pseudo-stochastic O matrix for a 2-qubit gate."""
    T_2q_rows = [np.conj(Pi_AB).flatten() for Pi_AB in projectors_2q]
    T_2q = np.vstack(T_2q_rows) * 0.25
    T_2q_inv = sla.inv(T_2q) 
    U_super = np.kron(U_2q, np.conj(U_2q))
    O_complex = T_2q @ U_super @ T_2q_inv
    return jnp.array(np.real(O_complex), dtype=jnp.float64)

# Base Gates
H_gate_np = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
S_Hadamard = get_1q_S_matrix_np(H_gate_np)
O_H_fwd = get_1q_S_matrix_np(sla.sqrtm(H_gate_np))
O_H_bwd = get_1q_S_matrix_np(np.conj(sla.sqrtm(H_gate_np).T))

def controlled_phased_gates(i, j):
    """Generates the specific controlled-phase rotation matrix for the QFT."""
    power = j - i
    angle = np.pi / (2**power)
    phase = np.exp(1j * angle)
    CP = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, phase]], dtype=np.complex128)
    return CP


# 2. EXACT 1D CLUSTER STATE MPS PROBABILITIES
@jax.jit
def compute_cluster_state_mps_probs(batch_a):
    """
    Computes exact POVM probabilities for a 1D Cluster State using a Matrix Product 
    State (MPS) contraction. This acts as the exact initialization target.
    """
    H_tensor = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128) / jnp.sqrt(2.0)
    
    def single_string_prob(a_string):
        val = SIC_PROJECTORS[a_string[0]]
        def body_fn(i, current_val):
            proj = SIC_PROJECTORS[a_string[i]]
            return jnp.matmul(H_tensor, jnp.matmul(current_val * proj, H_tensor))
        final_val = jax.lax.fori_loop(1, num_qubits, body_fn, val)
        return jnp.real(jnp.trace(final_val)) * (0.5 ** num_qubits)
        
    return jax.vmap(single_string_prob)(batch_a)


# 3. QUANTUM TRANSFORMER ARCHITECTURE
# The autoregressive Quantum Transformer. A strict causal mask ensures the  
# prediction for qubit k attends exclusively to qubits < k.
class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    @nn.compact
    def __call__(self, x, mask):
        attn_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.d_model)(x, x, mask=mask)
        x = nn.LayerNorm()(x + attn_out)
        ff_out = nn.Dense(features=self.d_model * 2)(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(features=self.d_model)(ff_out)
        x = nn.LayerNorm()(x + ff_out)
        return x

class QuantumTransformer(nn.Module):
    max_qubits: int 
    d_model: int = 64
    num_heads: int = 8
    num_layers: int = 4
    @nn.compact
    def __call__(self, x):
        batch_size, seq_len = x.shape
        start_tokens = jnp.full((batch_size, 1), 4, dtype=jnp.int32)
        x = jnp.concatenate([start_tokens, x[:, :-1]], axis=1)
        x_emb = nn.Embed(num_embeddings=5, features=self.d_model)(x)
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02), (1, self.max_qubits, self.d_model))
        x_emb = x_emb + pos_emb[:, :seq_len, :]
        mask = nn.make_causal_mask(x)
        h = x_emb
        for _ in range(self.num_layers):
            h = TransformerBlock(d_model=self.d_model, num_heads=self.num_heads)(h, mask)
        logits = nn.Dense(features=4)(h)
        return nn.log_softmax(logits, axis=-1)

def get_model_log_prob(logits_out, a_string):
    selected_log_probs = jnp.take_along_axis(logits_out, a_string[..., None], axis=-1).squeeze(-1)
    return jnp.sum(selected_log_probs, axis=-1)

# Marginalization helpers to compute targets efficiently
LOCAL_STATES_1Q = jnp.arange(4, dtype=jnp.int32)
def compute_single_target_1q(old_params, single_a, O_matrix, q_target):
    a_primes = jnp.tile(single_a, (4, 1)).at[:, q_target].set(LOCAL_STATES_1Q)
    probs_a_primes = jnp.exp(get_model_log_prob(model.apply(old_params, a_primes), a_primes))
    return jnp.dot(O_matrix[single_a[q_target], :], probs_a_primes)
compute_batch_targets_1q = jax.vmap(compute_single_target_1q, in_axes=(None, 0, None, None))

LOCAL_STATES_2Q = jnp.array(list(itertools.product([0, 1, 2, 3], repeat=2)), dtype=jnp.int32)
def compute_single_target_2q(old_params, single_a, O_matrix, q1, q2):
    a_primes = jnp.tile(single_a, (16, 1))
    a_primes = a_primes.at[:, q1].set(LOCAL_STATES_2Q[:, 0]).at[:, q2].set(LOCAL_STATES_2Q[:, 1])
    probs_a_primes = jnp.exp(get_model_log_prob(model.apply(old_params, a_primes), a_primes))
    return jnp.dot(O_matrix[single_a[q1] * 4 + single_a[q2], :], probs_a_primes)
compute_batch_targets_2q = jax.vmap(compute_single_target_2q, in_axes=(None, 0, None, None, None))

# --- Update Steps & Hybrid Losses ---
# Hybrid Loss: Combines Kullback-Leibler (KL) divergence penalty with a 
# Forward-Backward symmetry penalty to prevent representation collapse and maintain unitarity.
def cluster_init_loss_fn(params, batch_a, target_probs):
    log_probs_new = get_model_log_prob(model.apply(params, batch_a), batch_a)
    ratios_kl = jnp.exp(jnp.clip(jnp.log(jnp.clip(target_probs, a_min=1e-250)) - log_probs_new, -10.0, 10.0))
    return -jnp.mean(jax.lax.stop_gradient(ratios_kl - jnp.mean(ratios_kl)) * log_probs_new)

@functools.partial(jax.jit, static_argnames=['optimizer'])
def update_step_cluster_init(opt_state, params, batch_a, target_probs, optimizer):
    loss, grads = jax.value_and_grad(cluster_init_loss_fn)(params, batch_a, target_probs)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

def hybrid_loss_fn_1q(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q_target):
    log_probs_new = get_model_log_prob(model.apply(new_params, batch_a), batch_a)
    targets_full = compute_batch_targets_1q(old_params, batch_a, O_U, q_target)
    ratios_kl = jnp.exp(jnp.clip(jnp.log(jnp.clip(targets_full, a_min=1e-250)) - log_probs_new, -10.0, 10.0))
    loss_kl = -jnp.mean(jax.lax.stop_gradient(ratios_kl - jnp.mean(ratios_kl)) * log_probs_new)
    targets_halfway_fwd = compute_batch_targets_1q(old_params, batch_a, O_fwd, q_target)
    preds_halfway_bwd = compute_batch_targets_1q(new_params, batch_a, O_bwd, q_target)
    return loss_kl + jnp.mean((targets_halfway_fwd - preds_halfway_bwd)**2)

@functools.partial(jax.jit, static_argnames=['q_target', 'optimizer'])
def update_step_1q(opt_state, new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q_target, optimizer):
    loss, grads = jax.value_and_grad(hybrid_loss_fn_1q)(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q_target)
    updates, opt_state = optimizer.update(grads, opt_state, new_params)
    return optax.apply_updates(new_params, updates), opt_state, loss

def hybrid_loss_fn_2q(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2):
    log_probs_new = get_model_log_prob(model.apply(new_params, batch_a), batch_a)
    targets_full = compute_batch_targets_2q(old_params, batch_a, O_U, q1, q2)
    ratios_kl = jnp.exp(jnp.clip(jnp.log(jnp.clip(targets_full, a_min=1e-250)) - log_probs_new, -10.0, 10.0))
    loss_kl = -jnp.mean(jax.lax.stop_gradient(ratios_kl - jnp.mean(ratios_kl)) * log_probs_new)
    targets_halfway_fwd = compute_batch_targets_2q(old_params, batch_a, O_fwd, q1, q2)
    preds_halfway_bwd = compute_batch_targets_2q(new_params, batch_a, O_bwd, q1, q2)
    return loss_kl + jnp.mean((targets_halfway_fwd - preds_halfway_bwd)**2)

@functools.partial(jax.jit, static_argnames=['q1', 'q2', 'optimizer'])
def update_step_2q(opt_state, new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2, optimizer):
    loss, grads = jax.value_and_grad(hybrid_loss_fn_2q)(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2)
    updates, opt_state = optimizer.update(grads, opt_state, new_params)
    return optax.apply_updates(new_params, updates), opt_state, loss

@functools.partial(jax.jit, static_argnames=['batch_size', 'num_qubits'])
def sample_from_model(params, rng_key, batch_size, num_qubits):
    init_samples = jnp.zeros((batch_size, num_qubits), dtype=jnp.int32)
    def sample_step(i, carry):
        samples, current_rng = carry
        current_rng, step_rng = jax.random.split(current_rng)
        log_probs_out = model.apply(params, samples)
        sampled_outcomes = jax.random.categorical(step_rng, log_probs_out[:, i, :], axis=-1).astype(jnp.int32)
        samples = samples.at[:, i].set(sampled_outcomes)
        return samples, current_rng
    final_samples, _ = jax.lax.fori_loop(0, num_qubits, sample_step, (init_samples, rng_key))
    return final_samples

# --- LOG-SPACE Monte Carlo Fidelity Checkers ---
# Fidelity evaluates the mixture distribution via AM-GM bounding to maintain numerical stability
@functools.partial(jax.jit, static_argnames=['batch_size', 'q_target'])
def check_step_fidelity_1q(new_params, old_params, rng, batch_size, O_U, q_target):
    batch_a = sample_from_model(new_params, rng, batch_size, num_qubits)
    log_P_new = get_model_log_prob(model.apply(new_params, batch_a), batch_a)
    
    P_target = compute_batch_targets_1q(old_params, batch_a, O_U, q_target)
    log_P_target = jnp.log(jnp.clip(P_target, a_min=1e-250))
    
    log_ratio = jnp.clip(log_P_target - log_P_new, a_min=-20.0, a_max=20.0)
    return jnp.mean(jnp.exp(0.5 * log_ratio))

@functools.partial(jax.jit, static_argnames=['batch_size', 'q1', 'q2'])
def check_step_fidelity_2q(new_params, old_params, rng, batch_size, O_U, q1, q2):
    batch_a = sample_from_model(new_params, rng, batch_size, num_qubits)
    log_P_new = get_model_log_prob(model.apply(new_params, batch_a), batch_a)
    
    P_target = compute_batch_targets_2q(old_params, batch_a, O_U, q1, q2)
    log_P_target = jnp.log(jnp.clip(P_target, a_min=1e-250))
    
    log_ratio = jnp.clip(log_P_target - log_P_new, a_min=-20.0, a_max=20.0)
    return jnp.mean(jnp.exp(0.5 * log_ratio))


# 4. INITIALIZATION & CHECKPOINT RESTORATION
log_print(f"\nInitializing {num_qubits}-Qubit QFT Runner...")
batch_size = 2048 
rng = jax.random.PRNGKey(42)

model = QuantumTransformer(max_qubits=num_qubits)
dummy_params = model.init(rng, jnp.zeros((batch_size, num_qubits), dtype=jnp.int32))

ckpt_dir = pathlib.Path(f"qft_{num_qubits}_fixed_checkpoints").resolve()
options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
ckpt_manager = ocp.CheckpointManager(ckpt_dir, options=options)

latest_step = ckpt_manager.latest_step()
gates_completed = 0

if latest_step is not None:
    log_print(f"Found existing checkpoint! Resuming from step checkpoint: {latest_step}")
    target_state = {'params': dummy_params, 'gates_completed': 0}
    restored = ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore(target_state))
    params = restored['params']
    gates_completed = restored['gates_completed']
else:
    log_print(f"No checkpoint found. Phase 1: Training Transformer to fit the {num_qubits}-Qubit Cluster State via MPS...")
    params = dummy_params
    step, success_streak = 0, 0
    pbar = tqdm(desc="Cluster State Fit", total=50000, file=log_file)
    
    # Isolated LR schedule for state setup
    lr_schedule = optax.exponential_decay(init_value=0.001, transition_steps=3000, decay_rate=0.85)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)
    
    while step < 50000:
        rng, step_rng = jax.random.split(rng)
        batch_a = sample_from_model(params, step_rng, batch_size, num_qubits)
        target_probs = compute_cluster_state_mps_probs(batch_a)
        
        params, opt_state, loss = update_step_cluster_init(opt_state, params, batch_a, target_probs, optimizer)
        current_loss = float(loss)
        
        if abs(current_loss) < 1e-4:
            success_streak += 1
        else:
            success_streak = 0
            
        if step % 2 == 0:
            with open(f"qft_{num_qubits}_loss.txt", "a") as f:
                f.write(f"Cluster_Init,{step},{current_loss:.8f}\n")
                
        pbar.update(1)
        step += 1
        
        if success_streak >= 2:
            log_print(f"Cluster State converged beautifully at step {step} with loss: {current_loss:.8f}")
            break
            
    pbar.close()
    ckpt_manager.save(0, args=ocp.args.StandardSave({'params': params, 'gates_completed': 0}), force=True)
    ckpt_manager.wait_until_finished()


# 5. STEP-WISE QFT EXECUTION LOOP
log_print("\nPhase 2: Executing Step-wise QFT Circuit Tensors...")
gate_counter = 0

for j in range(num_qubits):
    log_print(f"\n--- Processing QFT Layer Block on Qubit {j} ---")
    
    # Gate A: Hadamard Gate on Qubit j
    gate_counter += 1
    if gate_counter <= gates_completed:
        continue
        
    log_print(f"Applying Hadamard to Qubit {j}...")
    old_params = params.copy()
    step, success_streak = 0, 0
    
    # Isolated learning rate reset for this gate
    lr_schedule = optax.exponential_decay(init_value=0.001, transition_steps=5000, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)
    
    pbar = tqdm(desc=f"H({j})", total=30000, file=log_file)
    while step < 30000:
        rng, step_rng = jax.random.split(rng)
        batch_a = sample_from_model(params, step_rng, batch_size, num_qubits)
        params, opt_state, loss = update_step_1q(opt_state, params, old_params, batch_a, S_Hadamard, O_H_fwd, O_H_bwd, j, optimizer)
        
        if abs(float(loss)) < 1e-4:
            success_streak += 1
        else:
            success_streak = 0
            
        if step % 2 == 0:
            with open(f"qft_{num_qubits}_loss.txt", "a") as f:
                f.write(f"H({j}),{step},{float(loss):.8f}\n")
                
        pbar.update(1)
        step += 1
        if success_streak >= 2: break
    pbar.close()
    
    rng, rng_fid = jax.random.split(rng)
    fid = check_step_fidelity_1q(params, old_params, rng_fid, batch_size, S_Hadamard, j)
    log_print(f"-> Gate {gate_counter} [H({j})] Monte Carlo Transition Fidelity: {fid:.5f}")
    with open(f"qft_{num_qubits}_fidelity_tracker.txt", "a") as f: 
        f.write(f"{gate_counter},H({j}),{fid:.8f}\n")
    
    ckpt_manager.save(gate_counter, args=ocp.args.StandardSave({'params': params, 'gates_completed': gate_counter}), force=True)
    ckpt_manager.wait_until_finished()

    # Gate B: Controlled Phase Gates between i and j
    for i in range(j):
        gate_counter += 1
        if gate_counter <= gates_completed:
            continue
            
        log_print(f"Applying Controlled-Phase CP({i} <-> {j})...")
        CP_np = controlled_phased_gates(i, j)
        O_CP = get_2q_O_matrix_np(CP_np)
        O_CP_fwd = get_2q_O_matrix_np(sla.sqrtm(CP_np))
        O_CP_bwd = get_2q_O_matrix_np(np.conj(sla.sqrtm(CP_np).T))
        
        old_params = params.copy()
        step, success_streak = 0, 0
        
        # Isolated learning rate reset for this gate
        lr_schedule = optax.exponential_decay(init_value=0.001, transition_steps=5000, decay_rate=0.5)
        optimizer = optax.adam(learning_rate=lr_schedule)
        opt_state = optimizer.init(params)
        
        pbar = tqdm(desc=f"CP({i}-{j})", total=30000, file=log_file)
        while step < 30000:
            rng, step_rng = jax.random.split(rng)
            batch_a = sample_from_model(params, step_rng, batch_size, num_qubits)
            params, opt_state, loss = update_step_2q(opt_state, params, old_params, batch_a, O_CP, O_CP_fwd, O_CP_bwd, i, j, optimizer)
            
            if abs(float(loss)) < 1e-4:
                success_streak += 1
            else:
                success_streak = 0
                
            if step % 2 == 0:
                with open(f"qft_{num_qubits}_loss.txt", "a") as f:
                    f.write(f"CP({i}_{j}),{step},{float(loss):.8f}\n")
                    
            pbar.update(1)
            step += 1
            if success_streak >= 2: break
        pbar.close()
        
        rng, rng_fid = jax.random.split(rng)
        fid = check_step_fidelity_2q(params, old_params, rng_fid, batch_size, O_CP, i, j)
        log_print(f"-> Gate {gate_counter} [CP({i}_{j})] Monte Carlo Transition Fidelity: {fid:.5f}")
        with open(f"qft_{num_qubits}_fidelity_tracker.txt", "a") as f: 
            f.write(f"{gate_counter},CP({i}_{j}),{fid:.8f}\n")
        
        ckpt_manager.save(gate_counter, args=ocp.args.StandardSave({'params': params, 'gates_completed': gate_counter}), force=True)
        ckpt_manager.wait_until_finished()
        
    jax.clear_caches()

log_file.close()
log_print(f"All Done! {num_qubits}-Qubit Cluster state fully transformed by QFT.")




