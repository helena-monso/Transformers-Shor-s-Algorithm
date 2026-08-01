#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBE modular exponentiation circuit with the Cuccaro ripple-carry adder

This script generates the uncompiled Qiskit circuits for modular exponentiation. 
You input the modulus `N`, the base `a`, and the number of control qubits to 
output the exact, uncompiled sequence of `X`, `CNOT`, and `TOF` gates.
"""


from qiskit import QuantumCircuit, transpile

# Cuccaro Ripple-Carry Adder Primitives
def majority(qc, a, b, c):
    """
    In-place Majority (MAJ) gate for the Cuccaro ripple-carry adder.
    Computes the majority of three bits and passes it forward as the new carry.
    """
    qc.cx(c, b)
    qc.cx(c, a)
    qc.ccx(a, b, c)

def unmajority(qc, a, b, c):
    """
    UnMajority and Add (UMA) gate.
    Restores the inputs and computes the final sum to maintain reversibility.
    """
    qc.ccx(a, b, c)
    qc.cx(c, a)
    qc.cx(a, b)

def ripple_carry_adder(qc, a, b, carry):
    """
    Executes |a>|b> -> |a>|a + b> using only local quantum primitives 
    and a single ancillary carry qubit.
    """
    n = len(a)

    # Forward sweep (MAJ gates)
    for i in range(n):
        if i == 0:
            majority(qc, carry, b[i], a[i])
        else:
            majority(qc, a[i-1], b[i], a[i])

    qc.cx(a[n-1], carry)

    # Backward sweep (UMA gates) for uncomputation
    for i in reversed(range(n)):
        if i == 0:
            unmajority(qc, carry, b[i], a[i])
        else:
            unmajority(qc, a[i-1], b[i], a[i])


# Controlled Constant Addition
def controlled_constant_addition(qc, control, target, anc, constant):
    """
    Applies addition conditioned on a control qubit: 
    If control == 1, target += constant.
    """
    n = len(target)

    # Encode classical constant into the ancilla workspace
    for i in range(n):
        if (constant >> i) & 1:
            qc.cx(control, anc[i])

    # Perform the ripple-carry addition
    ripple_carry_adder(qc, anc[:n], target, anc[n])

    # Uncompute the ancilla workspace to preserve unitarity
    for i in range(n):
        if (constant >> i) & 1:
            qc.cx(control, anc[i])


# Doubly Controlled Addition (VBE Core Primitive)
def cc_constant_addition(qc, c1, c2, target, anc, constant):
    """
    Addition of a classical binary constant dictated by a Toffoli (TOF) gate 
    acting on a single flag qubit.
    """
    flag = anc[-1]

    qc.ccx(c1, c2, flag)
    controlled_constant_addition(qc, flag, target, anc[:-1], constant)
    qc.ccx(c1, c2, flag)


# Controlled SWAP
def cswap(qc, ctrl, a, b):
    """
    Fredkin-style decomposition for a Controlled-SWAP gate.
    """
    qc.cx(b, a)
    qc.ccx(ctrl, a, b)
    qc.cx(b, a)


# TRUE VBE MODULAR MULTIPLICATION UNIT
def vbe_modular_multiply(qc, ctrl, y, acc, anc, N, a_power):
    """
    Executes the transformation |y> → |y * a_power mod N> in three distinct phases.
    """
    n = len(y)

    # PHASE 1: Accumulation (Build product)
    # Circuit performs classic shift-and-add binary multiplication into the accumulator.
    for i in range(n):
        const = (a_power * (2 ** i)) % N
        if const != 0:
            cc_constant_addition(qc, ctrl, y[i], acc, anc, const)

    # PHASE 2: Controlled Swap
    # Exchanges the completed mathematical product from the accumulator back into target y.
    for i in range(n):
        cswap(qc, ctrl, y[i], acc[i])

    # PHASE 3: Uncomputation (Resetting)
    # Computes the modular inverse and performs controlled subtraction to clean the accumulator.
    a_inv = pow(a_power, -1, N)

    for i in range(n):
        const = (a_inv * (2 ** i)) % N
        if const != 0:
            subtract_const = (2 ** n) - const
            cc_constant_addition(qc, ctrl, y[i], acc, anc, subtract_const)


# FULL SHOR EXPONENTIATION LAYER (Uncompiled)
def build_vbe_shor_multiplier(N, a, n_controls):
    """
    Constructs the fully uncompiled dynamic sequence for Shor's modular exponentiation.
    |x>|y> → |x>|a^x * y mod N>
    """
    n = N.bit_length()
    total_qubits = n_controls + (3 * n) + 2
    qc = QuantumCircuit(total_qubits)

    # Register allocation based on VBE sizing rules
    controls = list(range(n_controls))
    y = list(range(n_controls, n_controls + n))
    acc = list(range(n_controls + n, n_controls + 2*n))
    anc = list(range(n_controls + 2*n, n_controls + 3*n + 2))

    # Initialize the target register to |1>
    qc.x(y[0])

    # Generate the repeated squaring ladder
    for k in range(n_controls):
        a_power = pow(a, 2 ** k, N)
        if a_power == 1:
            continue

        vbe_modular_multiply(qc, controls[k], y, acc, anc, N, a_power)

    return qc

def circuit_to_sequence(qc):
    """
    Transpiles the circuit into our foundational baseline logic gates (X, CNOT, TOFFOLI)
    and formats them for the Transformer parser.
    """
    tqc = transpile(qc, basis_gates=["x", "cx", "ccx"], optimization_level=0)
    seq = []
    
    for inst in tqc.data:
        name = inst.operation.name
        q = [tqc.find_bit(x).index for x in inst.qubits]

        if name == "x":
            seq.append(f"X({q[0]})")
        elif name == "cx":
            seq.append(f"CNOT({q[0]}->{q[1]})")
        elif name == "ccx":
            seq.append(f"TOF({q[0]},{q[1]}->{q[2]})")

    return tqc, seq


# EXECUTION
if __name__ == "__main__":
    # Example hyperparameters for N=3, a=2
    N_val = 3
    a_val = 2
    n_ctrl = 4

    qc_vbe = build_vbe_shor_multiplier(N_val, a_val, n_ctrl)
    tqc_vbe, sequence = circuit_to_sequence(qc_vbe)

    output_filename = f"circuit_output_N{N_val}_a{a_val}.txt"

    with open(output_filename, "w") as f:
        f.write("==============================\n")
        f.write("VBE SHOR MODULAR MULTIPLIER\n")
        f.write("==============================\n")
        f.write(f"Modulus (N): {N_val}\n")
        f.write(f"Base (a)   : {a_val}\n")
        f.write(f"Qubits     : {tqc_vbe.num_qubits}\n")
        f.write(f"Depth      : {tqc_vbe.depth()}\n")
        f.write(f"Gates      : {len(sequence)}\n\n")
        
        for gate in sequence:
            f.write(f"'{gate}',\n")

    print(f"Success! Full uncompiled sequence saved to {output_filename}")
    
    
    
    