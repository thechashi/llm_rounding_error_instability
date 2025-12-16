
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_rotation_analysis_results(
    json_path="exp11_results.json", 
    output_dir="exp11_plots"
):
    """
    Loads results from the JSON file and creates plots for each 'jump'.

    Args:
        json_path (str): Path to the input JSON file.
        output_dir (str): Directory to save the output plots.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data from JSON
    with open(json_path, 'r') as f:
        all_results = json.load(f)

    # Generate a plot for each jump
    for i, jump_result in enumerate(all_results):
        jump_val = jump_result['jump']
        rotations = jump_result['rotations']
        
        # Extract data for plotting
        rotation_shifts = [r['rotation_shift'] for r in rotations]
        num_changes_layer0 = [r['layer0']['num_changes'] for r in rotations]
        num_changes_layer1 = [r['layer1']['num_changes'] for r in rotations]
        
        # Extract all changed indices for histograms
        all_changed_indices_layer0 = [idx for r in rotations for idx in r['layer0']['changed_indices']]
        all_changed_indices_layer1 = [idx for r in rotations for idx in r['layer1']['changed_indices']]
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'Rotation Analysis for Jump={jump_val}', fontsize=20)
        
        # Plot for Layer 0
        axes[0, 0].plot(rotation_shifts, num_changes_layer0, marker='o', linestyle='-', label='Layer 0 (Embedding Output)')
        axes[0, 0].set_title('Layer 0: Changed Indices vs. Rotation Shift', fontsize=16)
        axes[0, 0].set_xlabel('Rotation Shift (n)', fontsize=12)
        axes[0, 0].set_ylabel('Number of Changed Indices', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True)
        
        # Plot for Layer 1
        axes[0, 1].plot(rotation_shifts, num_changes_layer1, marker='x', linestyle='--', label='Layer 1 (Transformer Block 1 Output)')
        axes[0, 1].set_title('Layer 1: Changed Indices vs. Rotation Shift', fontsize=16)
        axes[0, 1].set_xlabel('Rotation Shift (n)', fontsize=12)
        axes[0, 1].set_ylabel('Number of Changed Indices', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True)
        
        # Histogram for Layer 0
        sns.histplot(all_changed_indices_layer0, bins=50, ax=axes[1, 0], kde=True)
        axes[1, 0].set_title('Layer 0: Histogram of Changed Indices', fontsize=16)
        axes[1, 0].set_xlabel('Index', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        
        # Histogram for Layer 1
        sns.histplot(all_changed_indices_layer1, bins=50, ax=axes[1, 1], kde=True)
        axes[1, 1].set_title('Layer 1: Histogram of Changed Indices', fontsize=16)
        axes[1, 1].set_xlabel('Index', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        
        # Adjust layout and save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_filename = os.path.join(output_dir, f"exp11_jump_{jump_val}_comprehensive_plot.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Comprehensive plot saved for jump={jump_val} at: {output_filename}")

if __name__ == "__main__":
    plot_rotation_analysis_results()
