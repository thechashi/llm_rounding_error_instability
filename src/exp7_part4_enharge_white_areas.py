"""
Recreate B/W heatmaps from saved .npz files with enlarged white pixel regions
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def recreate_bw_heatmap(npz_path, output_path=None, enlarge_pixels=1):
    """
    Load .npz file and recreate B/W heatmap with enlarged white pixel regions
    
    Args:
        npz_path: Path to .npz file containing grid, e1_values, e2_values
        output_path: Optional path to save output. If None, saves to same dir as input
        enlarge_pixels: Number of pixels to expand white regions
    """
    # Load data
    data = np.load(npz_path)
    grid = data['grid']
    e1_values = data['e1_values']
    e2_values = data['e2_values']
    
    # Print diagnostics
    print(f"\n=== Diagnostics for {Path(npz_path).name} ===")
    print(f"Grid shape: {grid.shape}")
    print(f"Grid min value: {grid.min()}")
    print(f"Grid max value: {grid.max()}")
    
    # Create binary grid: 1 where L1 >= L2, 0 where L1 < L2 (same as original)
    binary_grid = (grid >= 0).astype(int)
    
    positive_count = np.sum(binary_grid == 1)
    negative_count = np.sum(binary_grid == 0)
    print(f"Binary 1 (L1 >= L2): {positive_count}")
    print(f"Binary 0 (L1 < L2): {negative_count}")
    print(f"Total pixels: {binary_grid.size}")
    
    # IMPORTANT: In matplotlib's 'binary' colormap with vmin=0, vmax=1:
    # Value 0 = WHITE
    # Value 1 = BLACK
    # So white pixels are where binary_grid == 0 (i.e., grid < 0)
    
    print(f"\nIn 'binary' colormap:")
    print(f"WHITE regions: binary_grid == 0 (grid < 0) = {negative_count} pixels")
    print(f"BLACK regions: binary_grid == 1 (grid >= 0) = {positive_count} pixels")
    
    # Enlarge white regions (where binary_grid == 0) by dilating them
    if enlarge_pixels > 0:
        for iteration in range(enlarge_pixels):
            enlarged_grid = binary_grid.copy()
            for i in range(1, binary_grid.shape[0] - 1):
                for j in range(1, binary_grid.shape[1] - 1):
                    # If current pixel is white (binary_grid == 0)
                    if binary_grid[i, j] == 0:
                        # Make all neighbors in 3x3 region zero (white)
                        enlarged_grid[i-1:i+2, j-1:j+2] = 0
            binary_grid = enlarged_grid
    
    # Determine output path
    if output_path is None:
        npz_path = Path(npz_path)
        output_path = npz_path.parent / f"{npz_path.stem}_bw_enlarged.png"
    
    # Extract title from filename
    filename = Path(npz_path).stem
    title = filename.replace('logit_diff_', '').replace('_', ' ')
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.imshow(binary_grid.T, 
               extent=[e1_values[0], e1_values[-1], e2_values[0], e2_values[-1]], 
               origin='lower', cmap='binary', aspect='auto', vmin=0, vmax=1)
    plt.xlabel('e1', fontsize=16)
    plt.ylabel('e2', fontsize=16)
    plt.title(f'{title} (Binary: White = L1≥L2, Black = L1<L2)', fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=13)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def batch_recreate_bw_heatmaps(directory, enlarge_pixels=1):
    """
    Recreate B/W heatmaps for all .npz files in a directory
    
    Args:
        directory: Directory containing .npz files
        enlarge_pixels: Number of pixels to expand white regions
    """
    directory = Path(directory)
    npz_files = sorted(directory.glob('*.npz'))
    
    if not npz_files:
        print(f"No .npz files found in {directory}")
        return
    
    print(f"Found {len(npz_files)} .npz files")
    for npz_file in npz_files:
        print(f"Processing: {npz_file.name}")
        recreate_bw_heatmap(str(npz_file), enlarge_pixels=enlarge_pixels)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python recreate_bw_heatmaps_enlarged.py <npz_path_or_directory> [enlarge_pixels]")
        print("\nExamples:")
        print("  Single file (1 pixel enlargement):  python recreate_bw_heatmaps_enlarged.py results/logit_diff_1st_2nd.npz")
        print("  With 3 pixel enlargement: python recreate_bw_heatmaps_enlarged.py results/logit_diff_1st_2nd.npz 3")
        print("  Batch:        python recreate_bw_heatmaps_enlarged.py results/")
        sys.exit(1)
    
    path = sys.argv[1]
    enlarge_pixels = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    path_obj = Path(path)
    
    if path_obj.is_dir():
        batch_recreate_bw_heatmaps(path, enlarge_pixels=enlarge_pixels)
    elif path_obj.is_file() and path_obj.suffix == '.npz':
        recreate_bw_heatmap(path, enlarge_pixels=enlarge_pixels)
    else:
        print(f"Invalid path: {path}")
        sys.exit(1)
# """
# Recreate B/W heatmaps from saved .npz files
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def recreate_bw_heatmap(npz_path, output_path=None):
#     """
#     Load .npz file and recreate B/W heatmap
    
#     Args:
#         npz_path: Path to .npz file containing grid, e1_values, e2_values
#         output_path: Optional path to save output. If None, saves to same dir as input
#     """
#     # Load data
#     data = np.load(npz_path)
#     grid = data['grid']
#     e1_values = data['e1_values']
#     e2_values = data['e2_values']
    
#     # Create binary grid: 1 where L1 >= L2, 0 where L1 < L2
#     binary_grid = (grid >= 0).astype(int)
    
#     # Determine output path
#     if output_path is None:
#         npz_path = Path(npz_path)
#         output_path = npz_path.parent / f"{npz_path.stem}_bw_enlarged.png"
    
#     # Extract title from filename
#     filename = Path(npz_path).stem
#     title = filename.replace('logit_diff_', '').replace('_', ' ')
    
#     # Create plot
#     plt.figure(figsize=(10, 8))
#     plt.imshow(binary_grid.T, 
#                extent=[e1_values[0], e1_values[-1], e2_values[0], e2_values[-1]], 
#                origin='lower', cmap='binary', aspect='auto', vmin=0, vmax=1)
#     plt.xlabel('e1')
#     plt.ylabel('e2')
#     plt.title(f'{title} (Binary: White = L1≥L2, Black = L1<L2)')
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"Saved: {output_path}")
#     return output_path

# def batch_recreate_bw_heatmaps(directory):
#     """
#     Recreate B/W heatmaps for all .npz files in a directory
    
#     Args:
#         directory: Directory containing .npz files
#     """
#     directory = Path(directory)
#     npz_files = sorted(directory.glob('*.npz'))
    
#     if not npz_files:
#         print(f"No .npz files found in {directory}")
#         return
    
#     print(f"Found {len(npz_files)} .npz files")
#     for npz_file in npz_files:
#         print(f"Processing: {npz_file.name}")
#         recreate_bw_heatmap(str(npz_file))

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) < 2:
#         print("Usage: python recreate_bw_heatmaps.py <npz_path_or_directory>")
#         print("\nExamples:")
#         print("  Single file:  python recreate_bw_heatmaps.py results/logit_diff_1st_2nd.npz")
#         print("  Batch:        python recreate_bw_heatmaps.py results/")
#         sys.exit(1)
    
#     path = sys.argv[1]
#     path_obj = Path(path)
    
#     if path_obj.is_dir():
#         batch_recreate_bw_heatmaps(path)
#     elif path_obj.is_file() and path_obj.suffix == '.npz':
#         recreate_bw_heatmap(path)
#     else:
#         print(f"Invalid path: {path}")
#         sys.exit(1)