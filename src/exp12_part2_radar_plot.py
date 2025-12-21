"""
Experiment 12 Part 2: Radar Plot of Stability Boundary

Loads the npz file from exp12 and creates a radar plot showing the stability
boundary in the e1-e2 space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import sys


def load_exp12_data(npz_path):
    """Load data from exp12 npz file"""
    data = np.load(npz_path, allow_pickle=True)

    thetas = data['thetas']
    max_s_values = data['max_s_values']
    e1 = data['e1']
    e2 = data['e2']
    singular_values = data['singular_values']
    input_text = str(data['input_text'])

    print("="*80)
    print("LOADED DATA")
    print("="*80)
    print(f"Input text: {input_text}")
    print(f"Number of angles: {len(thetas)}")
    print(f"Singular values: σ₁={singular_values[0]:.6f}, σ₂={singular_values[1]:.6f}")
    print(f"Max s range: [{np.min(max_s_values):.6e}, {np.max(max_s_values):.6e}]")
    print("="*80)

    return thetas, max_s_values, e1, e2, singular_values, input_text


def create_radar_plot(thetas, max_s_values, input_text, save_dir):
    """Create radar plot (polar plot) of stability boundary"""

    # Convert to degrees for display
    degrees = np.degrees(thetas)

    print(f"  Plotting {len(thetas)} points")
    print(f"  Theta range: [{thetas[0]:.4f}, {thetas[-1]:.4f}] rad")
    print(f"  Max_s range: [{np.min(max_s_values):.4e}, {np.max(max_s_values):.4e}]")

    # Create figure with polar projection
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='polar')

    # Close the loop for plotting
    thetas_closed = np.append(thetas, thetas[0])
    max_s_closed = np.append(max_s_values, max_s_values[0])

    print(f"  After closing loop: {len(thetas_closed)} points")

    # Plot the boundary - use more points for smoother curve
    ax.plot(thetas_closed, max_s_closed, linewidth=2, color='#2E86AB',
            label='Stability Boundary', zorder=3, marker='', linestyle='-')
    ax.fill(thetas_closed, max_s_closed, alpha=0.4, color='#A23B72', zorder=2)

    # Mark cardinal directions
    cardinal_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    cardinal_labels = ['e₁', 'e₂', '-e₁', '-e₂']
    cardinal_colors = ['red', 'blue', 'red', 'blue']

    for angle, label, color in zip(cardinal_angles, cardinal_labels, cardinal_colors):
        idx = np.argmin(np.abs(thetas - angle))
        ax.plot(angle, max_s_values[idx], 'o', markersize=12,
                color=color, zorder=4, markeredgecolor='black', markeredgewidth=1.5)
        # Add text label
        ax.text(angle, max_s_values[idx] * 1.15, label,
                fontsize=14, fontweight='bold', color=color,
                ha='center', va='center')

    # Find min and max points
    min_idx = np.argmin(max_s_values)
    max_idx = np.argmax(max_s_values)

    ax.plot(thetas[min_idx], max_s_values[min_idx], 'v', markersize=14,
            color='green', label=f'Min: {max_s_values[min_idx]:.2e}',
            zorder=5, markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(thetas[max_idx], max_s_values[max_idx], '^', markersize=14,
            color='orange', label=f'Max: {max_s_values[max_idx]:.2e}',
            zorder=5, markeredgecolor='black', markeredgewidth=1.5)

    # Formatting
    ax.set_theta_zero_location('E')  # 0 degrees at East (e1 direction)
    ax.set_theta_direction(1)  # Counter-clockwise

    # Set title
    title = f'Stability Boundary Radar Plot\n"{input_text}"\n' + \
            f'Perturbation: s(cos(θ)·e₁ + sin(θ)·e₂)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)

    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), fontsize=11)

    # Grid
    ax.grid(True, alpha=0.5, linestyle='--', linewidth=1)

    # Set radial label
    ax.set_ylabel('Perturbation Magnitude (s)', labelpad=40, fontsize=12)

    plt.tight_layout()

    # Save
    radar_path = os.path.join(save_dir, "radar_plot.pdf")
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar plot (PDF): {radar_path}")

    radar_path_png = os.path.join(save_dir, "radar_plot.png")
    plt.savefig(radar_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved radar plot (PNG): {radar_path_png}")

    plt.show()
    plt.close()


def create_cartesian_2d_plot(thetas, max_s_values, e1, e2, input_text, save_dir):
    """
    Create 2D Cartesian plot in e1-e2 space

    For each theta and max_s, compute the actual point:
    point = max_s * (cos(theta)*e1 + sin(theta)*e2)

    Then plot the e1-component vs e2-component
    """

    print(f"  Plotting {len(thetas)} points in Cartesian e1-e2 space")

    # For each angle, compute the boundary point in the e1-e2 coordinate system
    # The perturbation is: s*(cos(theta)*e1 + sin(theta)*e2)
    # In the e1-e2 basis, this point has coordinates (s*cos(theta), s*sin(theta))

    x_coords = max_s_values * np.cos(thetas)  # e1 component
    y_coords = max_s_values * np.sin(thetas)  # e2 component

    print(f"  X range: [{np.min(x_coords):.4e}, {np.max(x_coords):.4e}]")
    print(f"  Y range: [{np.min(y_coords):.4e}, {np.max(y_coords):.4e}]")

    # Close the loop
    x_coords_closed = np.append(x_coords, x_coords[0])
    y_coords_closed = np.append(y_coords, y_coords[0])

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot boundary as line
    ax.plot(x_coords_closed, y_coords_closed, linewidth=2,
            color='#2E86AB', label='Stability Boundary (line)', zorder=3, alpha=0.7)
    ax.fill(x_coords_closed, y_coords_closed, alpha=0.2,
            color='#A23B72', zorder=2)

    # Add scatter plot of all points
    ax.scatter(x_coords, y_coords, c=max_s_values, cmap='viridis',
               s=20, alpha=0.8, edgecolors='black', linewidth=0.5,
               label='Data points', zorder=4)

    # Add colorbar for scatter points
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Max perturbation (s)')
    cbar.ax.tick_params(labelsize=10)

    # Mark cardinal directions
    cardinal_indices = []
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        idx = np.argmin(np.abs(thetas - angle))
        cardinal_indices.append(idx)

    cardinal_labels = ['e₁', 'e₂', '-e₁', '-e₂']
    cardinal_colors = ['red', 'blue', 'red', 'blue']

    for idx, label, color in zip(cardinal_indices, cardinal_labels, cardinal_colors):
        ax.plot(x_coords[idx], y_coords[idx], 'o', markersize=12,
                color=color, zorder=4, markeredgecolor='black', markeredgewidth=1.5)
        # Offset text label
        offset = 1.15
        ax.text(x_coords[idx] * offset, y_coords[idx] * offset, label,
                fontsize=14, fontweight='bold', color=color,
                ha='center', va='center')

    # Find and mark min/max
    min_idx = np.argmin(max_s_values)
    max_idx = np.argmax(max_s_values)

    ax.plot(x_coords[min_idx], y_coords[min_idx], 'v', markersize=14,
            color='green', label=f'Min radius: {max_s_values[min_idx]:.2e}',
            zorder=5, markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(x_coords[max_idx], y_coords[max_idx], '^', markersize=14,
            color='orange', label=f'Max radius: {max_s_values[max_idx]:.2e}',
            zorder=5, markeredgecolor='black', markeredgewidth=1.5)

    # Origin
    ax.plot(0, 0, 'ko', markersize=8, zorder=6, label='Origin')

    # Axes through origin
    max_extent = max(np.max(np.abs(x_coords)), np.max(np.abs(y_coords))) * 1.2
    ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5, zorder=1)
    ax.axvline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5, zorder=1)

    # Labels
    ax.set_xlabel('e₁ direction (perturbation component)', fontsize=14, fontweight='bold')
    ax.set_ylabel('e₂ direction (perturbation component)', fontsize=14, fontweight='bold')

    title = f'Stability Boundary in e₁-e₂ Space\n"{input_text}"\n' + \
            f'Each point: max_s × (cos(θ)·e₁ + sin(θ)·e₂)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)

    # Legend
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()

    # Save
    cartesian_path = os.path.join(save_dir, "cartesian_e1_e2_plot.pdf")
    plt.savefig(cartesian_path, dpi=300, bbox_inches='tight')
    print(f"Saved Cartesian e1-e2 plot (PDF): {cartesian_path}")

    cartesian_path_png = os.path.join(save_dir, "cartesian_e1_e2_plot.png")
    plt.savefig(cartesian_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved Cartesian e1-e2 plot (PNG): {cartesian_path_png}")

    plt.show()
    plt.close()


def create_scatter_plot(thetas, max_s_values, input_text, save_dir):
    """
    Create scatter plot in e1-e2 space with clear axis labels
    Center = origin (0,0)
    Right = +e1
    Left = -e1
    Top = +e2
    Bottom = -e2
    """
    print(f"  Creating scatter plot with {len(thetas)} points")

    # Compute coordinates
    x_coords = max_s_values * np.cos(thetas)  # e1 component
    y_coords = max_s_values * np.sin(thetas)  # e2 component

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))

    # Scatter plot with color mapping
    scatter = ax.scatter(x_coords, y_coords, c=np.degrees(thetas), cmap='hsv',
                        s=30, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Angle θ (degrees)')
    cbar.ax.tick_params(labelsize=11)

    # Mark origin
    ax.plot(0, 0, 'ko', markersize=12, zorder=10, label='Origin',
            markeredgecolor='white', markeredgewidth=2)

    # Draw axis lines
    max_extent = max(np.max(np.abs(x_coords)), np.max(np.abs(y_coords))) * 1.3
    ax.axhline(0, color='gray', linewidth=2, linestyle='--', alpha=0.6, zorder=1)
    ax.axvline(0, color='gray', linewidth=2, linestyle='--', alpha=0.6, zorder=1)

    # Mark and label cardinal directions on axes
    # Right: +e1
    ax.annotate('', xy=(max_extent*0.9, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='red', alpha=0.7))
    ax.text(max_extent*0.95, 0, '+e₁ →', fontsize=16, fontweight='bold',
            color='red', ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))

    # Left: -e1
    ax.annotate('', xy=(-max_extent*0.9, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='red', alpha=0.7))
    ax.text(-max_extent*0.95, 0, '← -e₁', fontsize=16, fontweight='bold',
            color='red', ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))

    # Top: +e2
    ax.annotate('', xy=(0, max_extent*0.9), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='blue', alpha=0.7))
    ax.text(0, max_extent*0.95, '+e₂\n↑', fontsize=16, fontweight='bold',
            color='blue', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', linewidth=2))

    # Bottom: -e2
    ax.annotate('', xy=(0, -max_extent*0.9), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='blue', alpha=0.7))
    ax.text(0, -max_extent*0.95, '↓\n-e₂', fontsize=16, fontweight='bold',
            color='blue', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', linewidth=2))

    # Labels and title
    ax.set_xlabel('e₁ direction →', fontsize=16, fontweight='bold')
    ax.set_ylabel('e₂ direction →', fontsize=16, fontweight='bold')

    title = f'Scatter Plot: Stability Boundary in e₁-e₂ Space\n"{input_text}"\n' + \
            f'{len(thetas)} points | Each point: max_s × (cos(θ)·e₁ + sin(θ)·e₂)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)

    # Set limits
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])

    plt.tight_layout()

    # Save
    scatter_path = os.path.join(save_dir, "scatter_e1_e2_plot.pdf")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot (PDF): {scatter_path}")

    scatter_path_png = os.path.join(save_dir, "scatter_e1_e2_plot.png")
    plt.savefig(scatter_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot (PNG): {scatter_path_png}")

    plt.show()
    plt.close()


def create_degree_plot(thetas, max_s_values, input_text, save_dir):
    """Create plot of max_s vs degrees"""
    degrees = np.degrees(thetas)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(degrees, max_s_values, linewidth=2, color='#2E86AB')
    ax.fill_between(degrees, max_s_values, alpha=0.4, color='#A23B72')

    # Mean line
    mean_s = np.mean(max_s_values)
    ax.axhline(y=mean_s, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_s:.2e}')

    # Mark cardinal directions
    cardinal_degs = [0, 90, 180, 270]
    cardinal_labels = ['e₁ (0°)', 'e₂ (90°)', '-e₁ (180°)', '-e₂ (270°)']

    for deg, label in zip(cardinal_degs, cardinal_labels):
        idx = np.argmin(np.abs(degrees - deg))
        ax.axvline(x=deg, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.plot(deg, max_s_values[idx], 'ro', markersize=8, zorder=5)
        ax.text(deg, max_s_values[idx] * 1.1, label,
                rotation=90, fontsize=10, ha='center', va='bottom')

    ax.set_xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Max Perturbation Magnitude (s)', fontsize=14, fontweight='bold')
    ax.set_title(f'Stability Boundary vs Angle\n"{input_text}"',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim([0, 360])

    plt.tight_layout()

    # Save
    deg_path = os.path.join(save_dir, "degrees_plot.pdf")
    plt.savefig(deg_path, dpi=300, bbox_inches='tight')
    print(f"Saved degrees plot (PDF): {deg_path}")

    deg_path_png = os.path.join(save_dir, "degrees_plot.png")
    plt.savefig(deg_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved degrees plot (PNG): {deg_path_png}")

    plt.show()
    plt.close()


def print_statistics(thetas, max_s_values):
    """Print detailed statistics"""
    degrees = np.degrees(thetas)

    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Mean max_s:        {np.mean(max_s_values):.6e}")
    print(f"Median max_s:      {np.median(max_s_values):.6e}")
    print(f"Std dev:           {np.std(max_s_values):.6e}")
    print(f"Min max_s:         {np.min(max_s_values):.6e}")
    print(f"Max max_s:         {np.max(max_s_values):.6e}")
    print(f"Range:             {np.ptp(max_s_values):.6e}")

    min_idx = np.argmin(max_s_values)
    max_idx = np.argmax(max_s_values)

    print(f"\nMin at: θ={thetas[min_idx]:.4f} rad ({degrees[min_idx]:.2f}°)")
    print(f"Max at: θ={thetas[max_idx]:.4f} rad ({degrees[max_idx]:.2f}°)")

    # Cardinal directions
    print("\nCardinal directions:")
    for angle, label in [(0, 'e₁'), (np.pi/2, 'e₂'), (np.pi, '-e₁'), (3*np.pi/2, '-e₂')]:
        idx = np.argmin(np.abs(thetas - angle))
        print(f"  {label:4s} (θ={angle:.4f} rad, {degrees[idx]:.1f}°): max_s = {max_s_values[idx]:.6e}")
    print("="*80)


def main(npz_path):
    """Main function"""
    # Load data
    thetas, max_s_values, e1, e2, singular_values, input_text = load_exp12_data(npz_path)
    print(thetas[:10], max_s_values[:10])
    # Print statistics
    print_statistics(thetas, max_s_values)

    # Get output directory (same as npz file directory)
    save_dir = os.path.dirname(npz_path)

    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    # Create plots
    print("\n[1/4] Creating radar plot...")
    create_radar_plot(thetas, max_s_values, input_text, save_dir)

    print("\n[2/4] Creating Cartesian e1-e2 plot (with scatter overlay)...")
    create_cartesian_2d_plot(thetas, max_s_values, e1, e2, input_text, save_dir)

    print("\n[3/4] Creating dedicated scatter plot...")
    create_scatter_plot(thetas, max_s_values, input_text, save_dir)

    print("\n[4/4] Creating degrees plot...")
    create_degree_plot(thetas, max_s_values, input_text, save_dir)

    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Output directory: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python exp12_part2_radar_plot.py <path_to_npz_file>")
        print("\nExample:")
        print("  python exp12_part2_radar_plot.py ../results/exp12_2025-12-18_10-00-00/polar_boundary_data.npz")
        sys.exit(1)

    npz_path = sys.argv[1]

    if not os.path.exists(npz_path):
        print(f"Error: File not found: {npz_path}")
        sys.exit(1)

    main(npz_path)

'''
python src/exp12_part2_radar_plot.py "/home/chashi/Desktop/Research/My Projects/results/exp12_2025-12-18_19-50-13/polar_boundary_data.npz"
'''