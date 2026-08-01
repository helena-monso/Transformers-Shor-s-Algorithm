#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncompiled Modular Exponentiation - VBE Architecture.
N=3, a=2 (Float32 Speed + Batch Accumulation Fidelity + Gate-by-Gate Checkpoints)

Simulates the dense uncompiled modular exponentiation phase of Shor's Algorithm.
"""

import os
import shutil
# MEMORY OPTIMIZATIONS
# Pre-allocation is disabled to prevent JAX from hoarding GPU memory, 
# allowing for exact math operations to run alongside the neural network.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
# jax.config.update("jax_enable_x64", True) # Kept off for Float32 Speed!
import jax.numpy as jnp
import flax.linen as nn
import optax
import functools
import itertools
import numpy as np
import scipy.linalg as sla
from tqdm.auto import tqdm
import math
import json
import orbax.checkpoint as ocp
import pathlib
import jax.tree_util


# REMOTE LOGGING SETUP
log_file = open("uncompiled_N3_a2.log", "w", encoding="utf-8")

def log_print(*args, **kwargs):
    print(*args, **kwargs) 
    print(*args, file=log_file, **kwargs)
    log_file.flush() 


# 1. THE UNCOMPILED VBE CIRCUITS
test_cases = [
    {
        "name": "VBE_N3_a2", 
        "N": 3, 
        "a": 2, 
        "n_controls": 4,
        "num_qubits": 12,
        "sequence": [
            'CNOT(8->6)', 'CNOT(8->10)', 'TOF(10,6->8)', 'TOF(0,4->11)', 
            'CNOT(11->9)', 'CNOT(9->7)', 'CNOT(9->8)', 'TOF(8,7->9)', 
            'CNOT(9->10)', 'TOF(8,7->9)', 'CNOT(9->8)', 'CNOT(8->7)', 
            'TOF(10,6->8)', 'CNOT(8->10)', 'CNOT(10->6)', 'CNOT(11->9)', 
            'TOF(0,4->11)', 'TOF(0,5->11)', 'CNOT(9->7)', 'CNOT(11->8)', 
            'CNOT(8->6)', 'CNOT(8->10)', 'TOF(10,6->8)', 'CNOT(9->8)', 
            'TOF(8,7->9)', 'CNOT(9->10)', 'TOF(8,7->9)', 'CNOT(9->8)', 
            'CNOT(8->7)', 'TOF(10,6->8)', 'CNOT(8->10)', 'CNOT(10->6)', 
            'CNOT(6->4)', 'CNOT(11->8)', 'TOF(0,5->11)', 'TOF(0,4->6)', 
            'CNOT(6->4)', 'CNOT(7->5)', 'TOF(0,5->7)', 'TOF(0,4->11)', 
            'CNOT(7->5)', 'CNOT(8->6)', 'CNOT(8->10)', 'TOF(10,6->8)', 
            'CNOT(11->9)', 'CNOT(9->7)', 'CNOT(9->8)', 'TOF(8,7->9)', 
            'CNOT(9->10)', 'TOF(8,7->9)', 'CNOT(9->8)', 'CNOT(8->7)', 
            'TOF(10,6->8)', 'CNOT(8->10)', 'CNOT(10->6)', 'CNOT(11->9)', 
            'TOF(0,4->11)', 'TOF(0,5->11)', 'CNOT(11->8)', 'CNOT(8->6)', 
            'CNOT(8->10)', 'TOF(10,6->8)', 'CNOT(11->9)', 'CNOT(9->7)', 
            'CNOT(9->8)', 'TOF(8,7->9)', 'CNOT(9->10)', 'TOF(8,7->9)', 
            'CNOT(9->8)', 'CNOT(8->7)', 'TOF(10,6->8)', 'CNOT(8->10)', 
            'CNOT(10->6)', 'CNOT(11->8)', 'CNOT(11->9)', 'TOF(0,5->11)'
        ]
    }
]


# 2. 64-BIT CPU MATRIX GENERATION (NumPy)
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

IC_PROJECTORS_2Q_list_np = [np.kron(Pi_A, Pi_B) for Pi_A in SIC_PROJECTORS_NP for Pi_B in SIC_PROJECTORS_NP]
IC_PROJECTORS_2Q_NP = np.array(IC_PROJECTORS_2Q_list_np)

IC_PROJECTORS_3Q_list_np = [np.kron(Pi_A, np.kron(Pi_B, Pi_C)) for Pi_A in SIC_PROJECTORS_NP for Pi_B in SIC_PROJECTORS_NP for Pi_C in SIC_PROJECTORS_NP]
IC_PROJECTORS_3Q_NP = np.array(IC_PROJECTORS_3Q_list_np)

def get_1q_S_matrix_np(U, projectors):
    num_outcomes = projectors.shape[0]
    d = int(np.sqrt(num_outcomes))
    S_matrix = np.zeros((num_outcomes, num_outcomes), dtype=np.float64)
    U_dag = np.conj(U.T)
    for i in range(num_outcomes):
        for j in range(num_outcomes):
            evolved_Pi_j = U @ projectors[j] @ U_dag
            s_ij = (1.0 / d) * np.real(np.trace(evolved_Pi_j @ projectors[i]))
            S_matrix[i, j] = (d + 1) * s_ij - (1.0 / d)
    return jnp.array(S_matrix, dtype=jnp.float64)

def get_2q_O_matrix_np(U_2q, projectors_2q=IC_PROJECTORS_2Q_NP):
    T_2q = np.vstack([np.conj(0.25 * Pi).flatten() for Pi in projectors_2q])
    T_2q_inv = sla.inv(T_2q) 
    U_super = np.kron(U_2q, np.conj(U_2q))
    return jnp.array(np.real(T_2q @ U_super @ T_2q_inv), dtype=jnp.float64)

def get_3q_O_matrix_np(U_3q, projectors_3q=IC_PROJECTORS_3Q_NP):
    """Computes the 3-qubit transition distributions natively to simulate Toffoli operations."""
    T_3q = np.vstack([np.conj(0.125 * Pi).flatten() for Pi in projectors_3q])
    T_3q_inv = sla.inv(T_3q) 
    U_super = np.kron(U_3q, np.conj(U_3q))
    return jnp.array(np.real(T_3q @ U_super @ T_3q_inv), dtype=jnp.float64)


# 3. UNITARIES 
H_gate = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
S_Hadamard = get_1q_S_matrix_np(H_gate, SIC_PROJECTORS_NP)
O_H_fwd = get_1q_S_matrix_np(sla.sqrtm(H_gate), SIC_PROJECTORS_NP)
O_H_bwd = get_1q_S_matrix_np(np.conj(sla.sqrtm(H_gate).T), SIC_PROJECTORS_NP)

X_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
S_X = get_1q_S_matrix_np(X_gate, SIC_PROJECTORS_NP)
O_X_fwd = get_1q_S_matrix_np(sla.sqrtm(X_gate), SIC_PROJECTORS_NP)
O_X_bwd = get_1q_S_matrix_np(np.conj(sla.sqrtm(X_gate).T), SIC_PROJECTORS_NP)

CNOT_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
O_CNOT = get_2q_O_matrix_np(CNOT_gate)
O_CNOT_fwd = get_2q_O_matrix_np(sla.sqrtm(CNOT_gate))
O_CNOT_bwd = get_2q_O_matrix_np(np.conj(sla.sqrtm(CNOT_gate).T))

U_TOFFOLI = np.eye(8, dtype=np.complex128)
U_TOFFOLI[6, 6] = 0; U_TOFFOLI[6, 7] = 1; U_TOFFOLI[7, 7] = 0; U_TOFFOLI[7, 6] = 1
O_TOFFOLI = get_3q_O_matrix_np(U_TOFFOLI)
O_TOFFOLI_fwd = get_3q_O_matrix_np(sla.sqrtm(U_TOFFOLI))
O_TOFFOLI_bwd = get_3q_O_matrix_np(np.conj(sla.sqrtm(U_TOFFOLI).T))

CS_DAG_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1j]], dtype=np.complex128)
phase_CS_DAG = np.exp(-1j * np.pi / 4)
V_CS_DAG = np.diag([1, 1, 1, phase_CS_DAG]).astype(np.complex128)
O_CS_DAG = get_2q_O_matrix_np(CS_DAG_gate)
O_CS_DAG_fwd = get_2q_O_matrix_np(V_CS_DAG)
O_CS_DAG_bwd = get_2q_O_matrix_np(np.conj(V_CS_DAG.T))


# 4. NEURAL NETWORK & LOG-SPACE LOSS FUNCTIONS
def get_initial_1q_distribution(projectors):
    rho_0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex64)
    P_exact_complex = jnp.array([0.5 * jnp.trace(rho_0 @ Pi) for Pi in projectors])
    return jnp.real(P_exact_complex)

SIC_PROJECTORS = jnp.array(SIC_PROJECTORS_NP, dtype=jnp.complex64)
P_initial = get_initial_1q_distribution(SIC_PROJECTORS)

# The autoregressive Quantum Transformer. A strict causal mask ensures the prediction 
# for qubit k attends exclusively to qubits < k.
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
        return nn.LayerNorm()(x + ff_out)

class QuantumTransformer(nn.Module):
    num_qubits: int
    d_model: int = 64
    num_heads: int = 8
    num_layers: int = 4
    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        start_tokens = jnp.full((batch_size, 1), 4, dtype=jnp.int32)
        x = jnp.concatenate([start_tokens, x[:, :-1]], axis=1)
        x_emb = nn.Embed(num_embeddings=5, features=self.d_model)(x)
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02), (1, self.num_qubits, self.d_model))
        h = x_emb + pos_emb
        mask = nn.make_causal_mask(x)
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

LOCAL_STATES_3Q = jnp.array(list(itertools.product([0, 1, 2, 3], repeat=3)), dtype=jnp.int32)
def compute_single_target_3q(old_params, single_a, O_matrix, q1, q2, q3):
    a_primes = jnp.tile(single_a, (64, 1))
    a_primes = a_primes.at[:, q1].set(LOCAL_STATES_3Q[:, 0]).at[:, q2].set(LOCAL_STATES_3Q[:, 1]).at[:, q3].set(LOCAL_STATES_3Q[:, 2])
    probs_a_primes = jnp.exp(get_model_log_prob(model.apply(old_params, a_primes), a_primes))
    return jnp.dot(O_matrix[single_a[q1] * 16 + single_a[q2] * 4 + single_a[q3], :], probs_a_primes)
compute_batch_targets_3q = jax.vmap(compute_single_target_3q, in_axes=(None, 0, None, None, None, None))

# --- Update Steps & Hybrid Losses ---
# Hybrid Loss: Combines Kullback-Leibler (KL) divergence penalty with a 
# Forward-Backward symmetry penalty to prevent representation collapse and maintain unitarity.
def case0_hybrid_loss_fn(params, batch_a, P_initial_1q, O_U, O_fwd, O_bwd, q_target):
    log_probs_out = model.apply(params, batch_a)
    log_probs_new = get_model_log_prob(log_probs_out, batch_a)
    P_target_q = jnp.dot(O_U, P_initial_1q)
    P_all_exact = P_initial_1q[batch_a].at[:, q_target].set(P_target_q[batch_a[:, q_target]])
    targets_exact = jnp.prod(P_all_exact, axis=1)
    log_targets = jnp.log(jnp.clip(targets_exact, a_min=1e-30))
    ratios_kl = jnp.exp(jnp.clip(log_targets - log_probs_new, -10.0, 10.0))
    loss_kl = -jnp.mean(jax.lax.stop_gradient(ratios_kl - jnp.mean(ratios_kl)) * log_probs_new)
    P_halfway_q = jnp.dot(O_fwd, P_initial_1q)
    P_all_halfway = P_initial_1q[batch_a].at[:, q_target].set(P_halfway_q[batch_a[:, q_target]])
    targets_halfway = jnp.prod(P_all_halfway, axis=1)
    preds_halfway = compute_batch_targets_1q(params, batch_a, O_bwd, q_target)
    return loss_kl + jnp.mean((targets_halfway - preds_halfway)**2)

@functools.partial(jax.jit, static_argnames=['q_target'])
def update_step_case0(opt_state, params, batch_a, P_initial_1q, O_U, O_fwd, O_bwd, q_target):
    loss, grads = jax.value_and_grad(case0_hybrid_loss_fn)(params, batch_a, P_initial_1q, O_U, O_fwd, O_bwd, q_target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

def caseN_hybrid_loss_fn_1q(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q_target):
    log_probs_out = model.apply(new_params, batch_a)
    log_probs_new = get_model_log_prob(log_probs_out, batch_a)
    targets_full = compute_batch_targets_1q(old_params, batch_a, O_U, q_target)
    log_targets = jnp.log(jnp.clip(targets_full, a_min=1e-30))
    ratios_kl = jnp.exp(jnp.clip(log_targets - log_probs_new, -10.0, 10.0))
    loss_kl = -jnp.mean(jax.lax.stop_gradient(ratios_kl - jnp.mean(ratios_kl)) * log_probs_new)
    targets_halfway = compute_batch_targets_1q(old_params, batch_a, O_fwd, q_target)
    preds_halfway = compute_batch_targets_1q(new_params, batch_a, O_bwd, q_target)
    return loss_kl + jnp.mean((targets_halfway - preds_halfway)**2)

@functools.partial(jax.jit, static_argnames=['q_target'])
def update_step_caseN_1q(opt_state, new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q_target):
    loss, grads = jax.value_and_grad(caseN_hybrid_loss_fn_1q)(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q_target)
    updates, opt_state = optimizer.update(grads, opt_state, new_params)
    return optax.apply_updates(new_params, updates), opt_state, loss

def caseN_hybrid_loss_fn_2q(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2):
    log_probs_out = model.apply(new_params, batch_a)
    log_probs_new = get_model_log_prob(log_probs_out, batch_a)
    targets_full = compute_batch_targets_2q(old_params, batch_a, O_U, q1, q2)
    log_targets = jnp.log(jnp.clip(targets_full, a_min=1e-30))
    ratios_kl = jnp.exp(jnp.clip(log_targets - log_probs_new, -10.0, 10.0))
    loss_kl = -jnp.mean(jax.lax.stop_gradient(ratios_kl - jnp.mean(ratios_kl)) * log_probs_new)
    targets_halfway = compute_batch_targets_2q(old_params, batch_a, O_fwd, q1, q2)
    preds_halfway = compute_batch_targets_2q(new_params, batch_a, O_bwd, q1, q2)
    return loss_kl + jnp.mean((targets_halfway - preds_halfway)**2)

@functools.partial(jax.jit, static_argnames=['q1', 'q2'])
def update_step_caseN_2q(opt_state, new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2):
    loss, grads = jax.value_and_grad(caseN_hybrid_loss_fn_2q)(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2)
    updates, opt_state = optimizer.update(grads, opt_state, new_params)
    return optax.apply_updates(new_params, updates), opt_state, loss

def caseN_hybrid_loss_fn_3q(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2, q3):
    log_probs_out = model.apply(new_params, batch_a)
    log_probs_new = get_model_log_prob(log_probs_out, batch_a)
    targets_full = compute_batch_targets_3q(old_params, batch_a, O_U, q1, q2, q3)
    log_targets = jnp.log(jnp.clip(targets_full, a_min=1e-30))
    ratios_kl = jnp.exp(jnp.clip(log_targets - log_probs_new, -10.0, 10.0))
    loss_kl = -jnp.mean(jax.lax.stop_gradient(ratios_kl - jnp.mean(ratios_kl)) * log_probs_new)
    targets_halfway = compute_batch_targets_3q(old_params, batch_a, O_fwd, q1, q2, q3)
    preds_halfway = compute_batch_targets_3q(new_params, batch_a, O_bwd, q1, q2, q3)
    return loss_kl + jnp.mean((targets_halfway - preds_halfway)**2)

@functools.partial(jax.jit, static_argnames=['q1', 'q2', 'q3'])
def update_step_caseN_3q(opt_state, new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2, q3):
    loss, grads = jax.value_and_grad(caseN_hybrid_loss_fn_3q)(new_params, old_params, batch_a, O_U, O_fwd, O_bwd, q1, q2, q3)
    updates, opt_state = optimizer.update(grads, opt_state, new_params)
    return optax.apply_updates(new_params, updates), opt_state, loss

@functools.partial(jax.jit, static_argnames=['batch_size', 'num_qubits'])
def sample_from_model(params, rng_key, batch_size, num_qubits):
    init_samples = jnp.zeros((batch_size, num_qubits), dtype=jnp.int32)
    def sample_step(i, carry):
        samples, current_rng = carry
        current_rng, step_rng = jax.random.split(current_rng)
        log_probs_out = model.apply(params, samples)
        sampled_outcomes = jax.random.categorical(step_rng, log_probs_out[:, i, :], axis=-1)
        samples = samples.at[:, i].set(sampled_outcomes)
        return samples, current_rng
    final_samples, _ = jax.lax.fori_loop(0, num_qubits, sample_step, (init_samples, rng_key))
    return final_samples


# 5. OOM-PROOF BRANCH TRACKER & BATCH-FIDELITY
# Explicit amplitude branch-tracking algorithm. Bypasses massive 2^N state-vector 
# allocations by recursively updating only non-zero probability amplitudes.
def apply_X_to_branches(branches, target_q):
    for b, amp in branches: b[target_q] = 1 - b[target_q]
    return branches

def apply_CNOT_to_branches(branches, control_q, target_q):
    for b, amp in branches:
        if b[control_q] == 1:
            b[target_q] = 1 - b[target_q]
    return branches

def apply_H_to_branches(branches, target_q):
    new_branches = []
    for b, amp in branches:
        b0 = b.copy(); b1 = b.copy()
        if b[target_q] == 0:
            b0[target_q] = 0; b1[target_q] = 1
            new_branches.extend([(b0, amp / np.sqrt(2)), (b1, amp / np.sqrt(2))])
        else:
            b0[target_q] = 0; b1[target_q] = 1
            new_branches.extend([(b0, amp / np.sqrt(2)), (b1, -amp / np.sqrt(2))])
    return new_branches

def apply_TOF_to_branches(branches, c1, c2, target_q):
    for b, amp in branches:
        if b[c1] == 1 and b[c2] == 1:
            b[target_q] = 1 - b[target_q]
    return branches

def apply_CS_DAG_to_branches(branches, c, t):
    new_branches = []
    for b, amp in branches:
        if b[c] == 1 and b[t] == 1:
            new_branches.append((b, amp * (-1j)))
        else:
            new_branches.append((b, amp))
    return new_branches

def aggregate_branches(branches):
    state_dict = {}
    for b, amp in branches:
        b_tuple = tuple(int(x) for x in b)
        if b_tuple in state_dict:
            state_dict[b_tuple] += amp
        else:
            state_dict[b_tuple] = amp
            
    new_branches = []
    for b_tuple, amp in state_dict.items():
        if np.abs(amp) > 1e-12: 
            new_branches.append((np.array(b_tuple, dtype=np.int32), amp))
    return new_branches

@functools.partial(jax.jit, static_argnames=['batch_size', 'num_qubits'])
def _compute_fidelity_math_batch(params, rng_key, batch_size, num_qubits, branch_bits, branch_coeffs):
    """
    Evaluates fidelity using a variance-stabilized AM-GM mixed Monte Carlo formulation
    to prevent numerical underflow in high-dimensional probability spaces.
    """
    batch_a = sample_from_model(params, rng_key, batch_size, num_qubits)
    log_P_model_batch = get_model_log_prob(model.apply(params, batch_a), batch_a)
    
    Pi_tensor = jnp.array(SIC_PROJECTORS_NP, dtype=jnp.complex128)
    
    def get_amplitude_prod(b_left, b_right):
        bl = jnp.broadcast_to(b_left, batch_a.shape)
        br = jnp.broadcast_to(b_right, batch_a.shape)
        return jnp.prod(Pi_tensor[batch_a, br, bl], axis=1)

    trace_val = jnp.zeros(batch_a.shape[0], dtype=jnp.complex128)
    for b1, c1 in zip(branch_bits, branch_coeffs):
        for b2, c2 in zip(branch_bits, branch_coeffs):
            coeff = c1 * jnp.conj(c2)
            trace_val += coeff * get_amplitude_prod(b1, b2)
            
    P_exact_batch = (0.5 ** num_qubits) * jnp.real(trace_val)
    is_non_zero = P_exact_batch > 1e-25
    log_exact = jnp.log(jnp.clip(P_exact_batch, a_min=1e-35))
    
    log_ratios = jnp.clip(log_exact - log_P_model_batch, a_min=-10.0, a_max=10.0)
    ratios_non_zero = jnp.exp(log_ratios)
    ratios = jnp.where(is_non_zero, ratios_non_zero, 0.0)
    
    return jnp.mean(jnp.sqrt(ratios))

def check_phase_aware_fidelity(params, model, rng, batch_size, num_qubits, branches, step_name, num_batches=16):
    log_print(f"\nCalculating MC Fidelity after {step_name}:")
    collapsed_branches = aggregate_branches(branches)
    
    bits_list = [jnp.array(b, dtype=jnp.int32) for b, c in collapsed_branches]
    coeffs_list = [jnp.complex128(c) for b, c in collapsed_branches]
        
    keys = jax.random.split(rng, num_batches)
    expval = 0.0
    
    loop = tqdm(range(num_batches), desc="Fidelity MC Accumulation", leave=False)
    for i in loop:
        batch_fid = _compute_fidelity_math_batch(params, keys[i], batch_size, num_qubits, bits_list, coeffs_list)
        expval += float(batch_fid)
        loop.set_description(f"Fidelity MC Accumulation (Avg: {expval / (i + 1):.4f})")
        
    final_fidelity = expval / num_batches
    log_print(f"Classical Fidelity (Fc): {final_fidelity:.4f}")
    return float(final_fidelity)


# 6. MAIN PIPELINE
for case_idx, case in enumerate(test_cases):
    case_name = case['name']
    
    # --- CHECKPOINT SAFETY PROTOCOL ---
    # Wipe the old checkpoint folder to prevent Orbax StepAlreadyExistsError
    ckpt_path_str = f"checkpoints_{case_name}"
    if os.path.exists(ckpt_path_str):
        log_print(f"Wiping old checkpoints at {ckpt_path_str} to start fresh...")
        shutil.rmtree(ckpt_path_str)
        
    ckpt_dir = pathlib.Path(ckpt_path_str).resolve()
    # CRITICAL: max_to_keep=10000 ensures ALL gate parameters are saved for post-eval
    options = ocp.CheckpointManagerOptions(max_to_keep=10000, create=True)
    ckpt_manager = ocp.CheckpointManager(ckpt_dir, options=options) 
    global_gate_counter = 0

    log_print(f"\n{'='*50}\nStarting Full Shor Pipeline: {case_name} ({case['num_qubits']} Qubits)\n{'='*50}")
    
    with open(f"{case_name}_fidelity.txt", "w") as f:
        f.write("experiment,gate_index,step_name,classical_fidelity\n")
    with open(f"{case_name}_loss.txt", "w") as f:
        f.write("experiment,step_name,step,loss\n")
    
    N_qubits = case['num_qubits']
    batch_size = 1024 

    rng = jax.random.PRNGKey(42)
    model = QuantumTransformer(num_qubits=N_qubits)
    params = model.init(rng, jnp.zeros((batch_size, N_qubits), dtype=jnp.int32))
        
    # Initial global optimizer just to have something in memory
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    
    max_steps = 20000
    loss_threshold = 1e-4
    current_branches = [(np.zeros(N_qubits, dtype=np.int32), 1.0 + 0j)]
    
    # --- PIPELINE STEP 1: INITIALIZATION ---
    log_print("\n--- STEP 1: INITIALIZATION ---")
    target_q = case['n_controls']
    log_print(f"Applying X gate to q_{target_q} (Init Target)")
    global_gate_counter += 1
    
    fresh_schedule = optax.exponential_decay(init_value=0.001, transition_steps=5000, decay_rate=0.5, end_value=1e-5)
    optimizer = optax.adam(learning_rate=fresh_schedule)
    opt_state = optimizer.init(params)
    
    current_loss, step, patience, required_patience = float('inf'), 0, 0, 3
    pbar = tqdm(desc=f"Init X({target_q})", total=max_steps)
    
    while patience < required_patience and step < max_steps:
        rng, step_rng = jax.random.split(rng)
        batch_a = sample_from_model(params, step_rng, batch_size=batch_size, num_qubits=N_qubits)
        params, opt_state, loss = update_step_case0(opt_state, params, batch_a, P_initial, S_X, O_X_fwd, O_X_bwd, q_target=target_q)
        
        if abs(float(loss)) < loss_threshold:
            patience += 1
        else:
            patience = 0
            
        if step % 2 == 0:
            with open(f"{case_name}_loss.txt", "a") as f:
                f.write(f"{case_name},Init_X,{step},{loss:.8f}\n")
                
        current_loss = float(loss); pbar.update(1); step += 1
    pbar.close()
    
    current_branches = apply_X_to_branches(current_branches, target_q)
    rng, rng_fid = jax.random.split(rng)
    fid = check_phase_aware_fidelity(params, model, rng_fid, batch_size, N_qubits, current_branches, f"Init_X({target_q})")
    
    with open(f"{case_name}_fidelity.txt", "a") as f:
        f.write(f"{case_name},{global_gate_counter},Init_X,{fid:.10f}\n")
    ckpt_manager.save(global_gate_counter, args=ocp.args.StandardSave({'params': params, 'opt_state': opt_state}), force=True)
    ckpt_manager.wait_until_finished()

    # --- PIPELINE STEP 2: SUPERPOSITION ---
    log_print("\n--- STEP 2: SUPERPOSITION ---")
    for q in range(case['n_controls']):
        log_print(f"Applying Hadamard to q_{q}")
        global_gate_counter += 1
        
        fresh_schedule = optax.exponential_decay(init_value=0.001, transition_steps=5000, decay_rate=0.5, end_value=1e-5)
        optimizer = optax.adam(learning_rate=fresh_schedule)
        opt_state = optimizer.init(params)
        
        old_params = params.copy(); current_loss, step, patience = float('inf'), 0, 0
        pbar = tqdm(desc=f"H({q})", total=max_steps)
        
        while patience < required_patience and step < max_steps:
            rng, step_rng = jax.random.split(rng)
            batch_a = sample_from_model(params, step_rng, batch_size=batch_size, num_qubits=N_qubits)
            params, opt_state, loss = update_step_caseN_1q(opt_state, params, old_params, batch_a, S_Hadamard, O_H_fwd, O_H_bwd, q_target=q)
            
            if abs(float(loss)) < loss_threshold:
                patience += 1
            else:
                patience = 0
                
            if step % 2 == 0:
                with open(f"{case_name}_loss.txt", "a") as f:
                    f.write(f"{case_name},Super_H({q}),{step},{loss:.8f}\n")
                    
            current_loss = float(loss); pbar.update(1); step += 1
        pbar.close()
        current_branches = apply_H_to_branches(current_branches, q)
        
        rng, rng_fid = jax.random.split(rng)
        fid = check_phase_aware_fidelity(params, model, rng_fid, batch_size, N_qubits, current_branches, f"Super_H({q})")
        with open(f"{case_name}_fidelity.txt", "a") as f:
            f.write(f"{case_name},{global_gate_counter},Super_H({q}),{fid:.10f}\n")
        ckpt_manager.save(global_gate_counter, args=ocp.args.StandardSave({'params': params, 'opt_state': opt_state}), force=True)
        ckpt_manager.wait_until_finished()

    # --- PIPELINE STEP 3: UNCOMPILED MODULAR EXPONENTIATION ---
    # Dense arithmetic block execution. The network must track carry bits through localized 
    # MAJ/UMA layers and unweave entanglement during uncomputation.
    log_print("\n--- STEP 3: UNCOMPILED MODULAR EXPONENTIATION ---")
    for cmd in case['sequence']:
        cmd = cmd.strip()
        global_gate_counter += 1
        log_print(f"Applying {cmd}")
        
        fresh_schedule = optax.exponential_decay(init_value=0.001, transition_steps=5000, decay_rate=0.5, end_value=1e-5)
        optimizer = optax.adam(learning_rate=fresh_schedule)
        opt_state = optimizer.init(params)
        
        old_params = params.copy(); current_loss, step, patience, required_patience = float('inf'), 0, 0, 3
        pbar = tqdm(desc=cmd, total=max_steps)
        
        if cmd.startswith("X("):
            q_t = int(cmd[2:-1])
            while patience < required_patience and step < max_steps:
                rng, step_rng = jax.random.split(rng)
                batch_a = sample_from_model(params, step_rng, batch_size, N_qubits)
                params, opt_state, loss = update_step_caseN_1q(opt_state, params, old_params, batch_a, S_X, O_X_fwd, O_X_bwd, q_t)
                
                if abs(float(loss)) < loss_threshold: patience += 1
                else: patience = 0
                    
                if step % 2 == 0:
                    with open(f"{case_name}_loss.txt", "a") as f: f.write(f"{case_name},{cmd},{step},{loss:.8f}\n")
                        
                current_loss = float(loss); pbar.update(1); step += 1
            current_branches = apply_X_to_branches(current_branches, q_t)
            
        elif cmd.startswith("TOF("):
            parts = cmd[4:-1].split('->'); controls = parts[0].split(',')
            q_c1, q_c2 = int(controls[0].strip()), int(controls[1].strip())
            q_t = int(parts[1].strip())
            
            while patience < required_patience and step < max_steps:
                rng, step_rng = jax.random.split(rng)
                batch_a = sample_from_model(params, step_rng, batch_size, N_qubits)
                params, opt_state, loss = update_step_caseN_3q(opt_state, params, old_params, batch_a, O_TOFFOLI, O_TOFFOLI_fwd, O_TOFFOLI_bwd, q_c1, q_c2, q_t)
                
                if abs(float(loss)) < loss_threshold: patience += 1
                else: patience = 0
                    
                if step % 2 == 0:
                    with open(f"{case_name}_loss.txt", "a") as f: f.write(f"{case_name},{cmd},{step},{loss:.8f}\n")
                        
                current_loss = float(loss); pbar.update(1); step += 1
            current_branches = apply_TOF_to_branches(current_branches, q_c1, q_c2, q_t)
            
        elif cmd.startswith("CNOT("):
            parts = cmd[5:-1].split('->')
            q_c, q_t = int(parts[0].strip()), int(parts[1].strip())
            
            while patience < required_patience and step < max_steps:
                rng, step_rng = jax.random.split(rng)
                batch_a = sample_from_model(params, step_rng, batch_size, N_qubits)
                params, opt_state, loss = update_step_caseN_2q(opt_state, params, old_params, batch_a, O_CNOT, O_CNOT_fwd, O_CNOT_bwd, q_c, q_t)
                
                if abs(float(loss)) < loss_threshold: patience += 1
                else: patience = 0
                    
                if step % 2 == 0:
                    with open(f"{case_name}_loss.txt", "a") as f: f.write(f"{case_name},{cmd},{step},{loss:.8f}\n")
                        
                current_loss = float(loss); pbar.update(1); step += 1
            current_branches = apply_CNOT_to_branches(current_branches, q_c, q_t)
            
        pbar.close()
        
        rng, rng_fid = jax.random.split(rng)
        fid = check_phase_aware_fidelity(params, model, rng_fid, batch_size, N_qubits, current_branches, f"Gate_{global_gate_counter}_{cmd}")
        with open(f"{case_name}_fidelity.txt", "a") as f:
            f.write(f"{case_name},{global_gate_counter},{cmd},{fid:.6f}\n")
        ckpt_manager.save(global_gate_counter, args=ocp.args.StandardSave({'params': params, 'opt_state': opt_state}), force=True)
        ckpt_manager.wait_until_finished()

    # FINAL CLEANUP & METADATA
    log_print("\n--- MODULAR EXPONENTIATION COMPLETE ---")
    
    # Save the run metadata so you have a record of the architecture and sequence
    metadata_path = os.path.join(ckpt_dir, f"metadata_case_{case_idx}.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(case, metadata_file, indent=4)
        
    jax.clear_caches()

log_print("\nModular Exponentiation Experiment Complete!")
log_file.close()





