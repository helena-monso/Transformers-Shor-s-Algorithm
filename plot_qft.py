#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the Classical Fidelity of the 25-Qubit QFT sequence on a Cluster State.
Reads dynamically from the generated qft_25_fidelity_tracker.txt file.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


# 1. Setup Academic Plotting Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 5), 
    'figure.dpi': 150
})


# 2. Main Execution
if __name__ == "__main__":
    filepath = "qft_25_fidelity_tracker.txt"
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run qft_25.py first.")
    else:
        # Read the raw output from the simulation
        df = pd.read_csv(filepath)

        # Automatically extract the target qubit block from the gate name 
        # (e.g., 'CP(0_4)' -> '4' or 'H(5)' -> '5')
        df['target_block'] = df['gate_name'].str.extract(r'(\d+)\)$').astype(int)

        # Normalize the transition fidelity relative to the maximum observed
        max_fidelity = df['transition_fidelity'].max()
        df['classical_fidelity_norm'] = df['transition_fidelity'] / max_fidelity

        # Create a continuous absolute step index for the X-axis
        df['total_gate_index'] = range(1, len(df) + 1)


        # 3. Plotting
        fig, ax = plt.subplots()
        ax.set_ylim(0.98, 1.01)

        # Plot a continuous faint line showing the overall drop trajectory
        ax.plot(
            df['total_gate_index'], 
            df['classical_fidelity_norm'], 
            linestyle='-', 
            color='gray', 
            alpha=0.5, 
            zorder=1
        )

        # Scatter the points on top, color-coded by the target qubit block
        sc = ax.scatter(
            df['total_gate_index'], 
            df['classical_fidelity_norm'], 
            c=df['target_block'], 
            cmap='Spectral', 
            s=25, 
            zorder=2
        )

        # Formatting the axes
        ax.set_xlabel('Total Gates Applied (Cumulative Sequence)')
        ax.set_ylabel('Classical Fidelity ($F_C$)')

        # Add a colorbar to explain what the colors mean
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Target Qubit Block ($n$)')

        # Framing the plot slightly thicker to match the academic style
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        cbar.outline.set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig("plot_qft_25.pdf", format="pdf", bbox_inches="tight")
        print("Successfully generated plot_qft_25.pdf")
        plt.show()