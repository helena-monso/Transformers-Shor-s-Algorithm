#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the Classical Fidelity of the Uncompiled Modular Exponentiation sequences and whole Shor 15.
Reads dynamically from the generated _fidelity.txt files.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


# 1. Academic Plot Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (11, 5),
    'figure.dpi': 150
})


# 2. Helper Functions
def gate_family(step):
    """Classifies the gate string into its primitive family for color coding."""
    # Safety check: Ensure the step is a string before applying string methods
    if not isinstance(step, str): 
        return "Other"
    
    if step.startswith("TOF"): return "TOF"
    elif step.startswith("CNOT"): return "CNOT"
    elif step.startswith("IQFT_H"): return "H"
    elif step.startswith("IQFT_CS"): return "R"
    elif step.startswith("Init_X"): return "X"
    elif step.startswith("Super_H"): return "H"
    else: return "Other"


# 3. Main Plotting Function
def plot_fidelity_from_file(csv_filepath, title, output_pdf):
    if not os.path.exists(csv_filepath):
        print(f"File {csv_filepath} not found. Skipping...")
        return

    # Read Data
    df = pd.read_csv(csv_filepath)
    df["fidelity"] = df["classical_fidelity"]
    df["gate_number"] = range(len(df))
    # Convert step_name to string just to be absolutely safe
    df["gate_family"] = df["step_name"].astype(str).apply(gate_family)

    # Colors
    colors = {
        "X":    "#3A3A3A",      # charcoal
        "H":    "#F58518",      # orange
        "CNOT": "#54A24B",   # green
        "TOF":  "#E457AD",    # pink/magenta
        "R":    "#4C78A8"       # blue
    }

    fig, ax = plt.subplots()

    # Overall trajectory line
    ax.plot(
        df["gate_number"],
        df["fidelity"],
        color="gray",
        linewidth=1.2,
        alpha=0.35,
        zorder=1
    )

    # Scatter points grouped by gate family
    for gate_type in colors:
        subset = df[df["gate_family"] == gate_type]
        if len(subset) > 0:
            ax.scatter(
                subset["gate_number"],
                subset["fidelity"],
                color=colors[gate_type],
                label=gate_type,
                s=35,
                zorder=2
            )

    # Formatting
    ax.set_xlabel("Total Gates Applied (Cumulative Sequence)")
    ax.set_ylabel(r"Classical Fidelity ($F_C$)")
    ax.set_title(title)

    ax.set_ylim(
        df["fidelity"].min() - 0.08,
        df["fidelity"].max() + 0.01
    )

    # Frame styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Legend
    ax.legend(frameon=True, loc="lower left", ncol=2)

    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    print(f"Successfully generated {output_pdf}")
    plt.close()


# 4. EXECUTION
if __name__ == "__main__":
    
    # Define the files to process and their corresponding plot titles
    experiments = [
        {
            "file": "VBE_N3_a2_fidelity.txt",
            "title": r"Modular Exponentiation for $N=3$, $a=2$ (12-Qubit VBE Circuit)",
            "out": "plot_modexp_N3.pdf"
        },
        {
            "file": "VBE_N4_a3_fidelity.txt",
            "title": r"Modular Exponentiation for $N=4$, $a=3$ (17-Qubit VBE Circuit)",
            "out": "plot_modexp_N4.pdf"
        },
        {
            "file": "VBE_N8_a3_fidelity.txt",
            "title": r"Modular Exponentiation for $N=8$, $a=3$ (22-Qubit VBE Circuit)",
            "out": "plot_modexp_N8.pdf"
        },
        {
            "file": "shor_N15_a4_fidelity.txt",
            "title": r"Shor's Algorithm for $N=15$, $a=4$ (22-Qubit VBE Circuit)",
            "out": "plot_shor_N15.pdf"
        }
    ]

    for exp in experiments:
        plot_fidelity_from_file(exp["file"], exp["title"], exp["out"])
        
        
        
        
        
        
        