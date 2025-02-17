import numpy as np
import h5py

def convert_npy_to_h5(input_path, output_path, chunk_size=1000):
    """
    Convert NPY to HDF5 format using np.load instead of memmap.
    """
    # Open the input file in read mode
    data = np.load(input_path, mmap_mode='r')
    print(f"Input data shape: {data.shape}, dtype: {data.dtype}")
    
    with h5py.File(output_path, 'w') as f:
        # Create dataset with same dtype as input
        dset = f.create_dataset('data',
                              shape=data.shape,
                              dtype=data.dtype,  # Use same dtype as input
                              chunks=(114, 114, chunk_size, 1),  # Changed chunk shape
                              compression='gzip')  # Moderate compression
        
        # Write data in chunks
        for i in range(0, data.shape[2], chunk_size):
            end_idx = min(i + chunk_size, data.shape[2])
            print(f"Processing slice {i} to {end_idx}")
            
            # Read chunk using direct numpy slicing
            chunk = data[:, :, i:end_idx, :]
            
            # Write to HDF5
            dset[:, :, i:end_idx, :] = chunk
            
            # Optional: verify this chunk
            verify_chunk = dset[:, :, i:end_idx, :]
            if not np.array_equal(chunk, verify_chunk):
                print(f"Warning: Verification failed for chunk {i} to {end_idx}")
            
            f.flush()


if __name__ == '__main__':
    convert_npy_to_h5('simulations/rd_10k_res1_matrices.npy',
                     'simulations/rd_10k_res1_matrices.h5')