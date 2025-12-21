import numpy as np                                                                                                                                                     
data = np.load("/home/chashi/Desktop/Research/My Projects/results/exp12_2025-12-18_19-50-13/polar_boundary_data.npz")                                                  
max_s = data['max_s_values']                                                                                                                                           
print(f"Non-zero count: {np.count_nonzero(max_s)}/{len(max_s)}")                                                                                                       
print(f"Min: {np.min(max_s):.2e}, Max: {np.max(max_s):.2e}")                                                                                                           
print(f"Mean: {np.mean(max_s):.2e}, Median: {np.median(max_s):.2e}")                                                                                                   
print(f"\nFirst 10 thetas: {data['thetas'][:10]}")                                                                                                                     
print(f"First 10 max_s: {max_s[:10]}")

thetas = data['thetas']
output_data = np.column_stack((thetas, max_s))
np.savetxt('polar_boundary_data.csv', output_data, delimiter=',', header='thetas,max_s', comments='')
print("Data saved to polar_boundary_data.csv")      