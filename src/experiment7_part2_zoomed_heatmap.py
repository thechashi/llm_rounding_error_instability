import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_matrix(filename):
    """Load the saved matrix and axis values"""
    data = np.load(filename)
    grid = data['grid']
    e1_values = data['e1_values']
    e2_values = data['e2_values']
    return grid, e1_values, e2_values

def find_nearest_idx(array, value):
    """Find the index of the nearest value in array"""
    idx = (np.abs(array - value)).argmin()
    return idx

def zoom_and_plot(filename, e1_start, e1_end, e2_start, e2_end, output_prefix=None):
    """Load matrix and plot zoomed region"""
    # Load data
    grid, e1_values, e2_values = load_matrix(filename)
    
    # Find indices for the requested range
    e1_start_idx = find_nearest_idx(e1_values, e1_start)
    e1_end_idx = find_nearest_idx(e1_values, e1_end)
    e2_start_idx = find_nearest_idx(e2_values, e2_start)
    e2_end_idx = find_nearest_idx(e2_values, e2_end)
    
    # Ensure proper ordering
    if e1_start_idx > e1_end_idx:
        e1_start_idx, e1_end_idx = e1_end_idx, e1_start_idx
    if e2_start_idx > e2_end_idx:
        e2_start_idx, e2_end_idx = e2_end_idx, e2_start_idx
    
    # Extract zoomed region
    zoomed_grid = grid[e1_start_idx:e1_end_idx+1, e2_start_idx:e2_end_idx+1]
    zoomed_e1 = e1_values[e1_start_idx:e1_end_idx+1]
    zoomed_e2 = e2_values[e2_start_idx:e2_end_idx+1]
    
    # Get actual range
    actual_e1_start = zoomed_e1[0]
    actual_e1_end = zoomed_e1[-1]
    actual_e2_start = zoomed_e2[0]
    actual_e2_end = zoomed_e2[-1]
    
    print(f"Loaded: {filename}")
    print(f"Original grid shape: {grid.shape}")
    print(f"Requested e1 range: [{e1_start:.2e}, {e1_end:.2e}]")
    print(f"Actual e1 range: [{actual_e1_start:.2e}, {actual_e1_end:.2e}]")
    print(f"Requested e2 range: [{e2_start:.2e}, {e2_end:.2e}]")
    print(f"Actual e2 range: [{actual_e2_start:.2e}, {actual_e2_end:.2e}]")
    print(f"Zoomed grid shape: {zoomed_grid.shape}")
    
    # Set output filename
    if output_prefix is None:
        output_prefix = filename.replace('.npz', '_zoomed')
    
    # Plot color heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(zoomed_grid.T, 
               extent=[actual_e1_start, actual_e1_end, actual_e2_start, actual_e2_end], 
               origin='lower', cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='L1 - L2')
    plt.xlabel('e1')
    plt.ylabel('e2')
    plt.title(f'Zoomed: e1=[{actual_e1_start:.2e}, {actual_e1_end:.2e}], e2=[{actual_e2_start:.2e}, {actual_e2_end:.2e}]')
    color_filename = f"{output_prefix}_color.png"
    plt.savefig(color_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {color_filename}")
    
    # Plot black & white heatmap
    binary_grid = (zoomed_grid >= 0).astype(int)
    plt.figure(figsize=(10, 8))
    plt.imshow(binary_grid.T, 
               extent=[actual_e1_start, actual_e1_end, actual_e2_start, actual_e2_end], 
               origin='lower', cmap='binary', aspect='auto', vmin=0, vmax=1)
    plt.xlabel('e1')
    plt.ylabel('e2')
    plt.title(f'Zoomed (Binary): e1=[{actual_e1_start:.2e}, {actual_e1_end:.2e}], e2=[{actual_e2_start:.2e}, {actual_e2_end:.2e}]')
    bw_filename = f"{output_prefix}_bw.png"
    plt.savefig(bw_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bw_filename}")
    
    return zoomed_grid, zoomed_e1, zoomed_e2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zoom into saved logit difference matrices')
    parser.add_argument('filename', type=str, help='Path to .npz file')
    parser.add_argument('--e1_start', type=float, required=True, help='Start of e1 range')
    parser.add_argument('--e1_end', type=float, required=True, help='End of e1 range')
    parser.add_argument('--e2_start', type=float, required=True, help='Start of e2 range')
    parser.add_argument('--e2_end', type=float, required=True, help='End of e2 range')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output prefix for saved files (default: filename_zoomed)')
    
    args = parser.parse_args()
    
    zoom_and_plot(
        args.filename,
        args.e1_start,
        args.e1_end,
        args.e2_start,
        args.e2_end,
        args.output
    )
    
    print("\nDone!")

# Example usage:
# RSV1 and RSV2
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_2nd.npz" --e1_start=1e-7 --e1_end=5e-7 --e2_start=-6e-7 --e2_end=-1e-6 --output "RSV1and2"
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_2nd.npz" --e1_start=-7.5e-7 --e1_end=-5e-7 --e2_start=2.5e-7 --e2_end=5e-7 --output "RSV1and2_V2"
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_2nd.npz" --e1_start=-10e-7 --e1_end=-7.5e-7 --e2_start=5.0e-7 --e2_end=7.5e-7 --output "RSV1and2_V2_V3"


# RSV1 and RSV10
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_10th.npz" --e1_start=1e-7 --e1_end=5e-7 --e2_start=-6e-7 --e2_end=-1e-6 --output "RSV1and10"
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_10th.npz" --e1_start=-5e-7 --e1_end=-2.5e-7 --e2_start=5e-7 --e2_end=7.5e-7 --output "RSV1and10_V1"
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_10th.npz" --e1_start=-7.5e-7 --e1_end=-10e-7 --e2_start=-7.5e-7 --e2_end=-10e-7 --output "RSV1and10_V2"
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_10th.npz" --e1_start=-7.5e-7 --e1_end=-5e-7 --e2_start=-2.5e-7 --e2_end=-5e-7 --output "RSV1and10_V3"

# RSV1 and RSV4096
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_4096th.npz" --e1_start=1e-7 --e1_end=5e-7 --e2_start=-6e-7 --e2_end=-1e-6 --output "RSV1and4096"
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_4096th.npz" --e1_start=-6.5e-7 --e1_end=-4.5e-7 --e2_start=-6.5e-7 --e2_end=-4.5e-7 --output "RSV1and409_V1"
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_4096th.npz" --e1_start=-7.5e-7 --e1_end=-5e-7 --e2_start=2.5e-7 --e2_end=5.0e-7 --output "RSV1and4096_V2"
# python3 src/experiment7_part2_zoomed_map.py "/home/chashi/Desktop/Research/My Projects/llm_rounding_error_instability/logit_diff_1st_4096th.npz" --e1_start=-7.5e-7 --e1_end=-5e-7 --e2_start=7.5e-7 --e2_end=10e-7 --output "RSV1and4096_V3"


# python zoom_script.py logit_diff_1st_2nd.npz --e1_start -5e-7 --e1_end 5e-7 --e2_start -5e-7 --e2_end 5e-7
# python zoom_script.py logit_diff_1st_2nd.npz --e1_start 0 --e1_end 1e-6 --e2_start -1e-6 --e2_end 0 --output my_zoom