import tensorflow as tf

"""
  generalized sparse-to-sparse multiplication leveraging TF computational graphs and GPU optimizations
"""


# Define an input signature to create a single, reusable graph
SPGEMM_SIGNATURE = [
    tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
    tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32)
]

@tf.function(input_signature=SPGEMM_SIGNATURE)
def spgemm_robust_aggregation(A_sp: tf.SparseTensor, B_sp: tf.SparseTensor) -> tf.SparseTensor:
    """
    Performs SpGEMM using a robust aggregation method that explicitly
    sums values for duplicate indices before final tensor construction.
    """
    # Steps 1-3 are the same: Find all products and their destination coordinates
    A_indices, A_values = A_sp.indices, A_sp.values
    B_indices, B_values = B_sp.indices, B_sp.values
    A_i, A_k = A_indices[:, 0], A_indices[:, 1]
    B_k, B_j = B_indices[:, 0], B_indices[:, 1]
    
    matching_k = tf.equal(A_k[:, None], B_k[None, :])
    match_indices = tf.where(matching_k)
    A_match_idx, B_match_idx = match_indices[:, 0], match_indices[:, 1]

    all_C_i = tf.gather(A_i, A_match_idx)
    all_C_j = tf.gather(B_j, B_match_idx)
    all_C_values = tf.gather(A_values, A_match_idx) * tf.gather(B_values, B_match_idx)

    # 4. Explicitly sum values for duplicate indices (The Robust Aggregation)
    # Create a unique 1D integer ID for each (i, j) coordinate pair.
    output_num_cols = tf.cast(B_sp.dense_shape[1], tf.int64)
    segment_ids_flat = all_C_i * output_num_cols + all_C_j
    
    # Use tf.unique to find the final unique output coordinates and the mapping.
    unique_flat_indices, segment_mapping = tf.unique(segment_ids_flat)

    # Use segment_sum to sum the values that belong to the same coordinate.
    summed_values = tf.math.segment_sum(all_C_values, segment_mapping)

    # 5. Assemble the final SparseTensor from the unique indices and summed values.
    # Convert the unique 1D IDs back to [row, col] format.
    final_C_i = unique_flat_indices // output_num_cols
    final_C_j = unique_flat_indices % output_num_cols
    final_C_indices = tf.stack([final_C_i, final_C_j], axis=1)
    
    output_shape = (A_sp.dense_shape[0], B_sp.dense_shape[1])
    
    return tf.SparseTensor(
        indices=final_C_indices,
        values=summed_values,
        dense_shape=tf.cast(output_shape, tf.int64)
    )

if __name__ == "__main__":
    # --- How to use it ---
    A = tf.sparse.from_dense(tf.constant([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]], dtype=tf.float32))
    B = tf.sparse.from_dense(tf.constant([[5.0, 6.0], [7.0, 0.0], [0.0, 8.0]], dtype=tf.float32))

    # Call the new, robust function
    C = spgemm_robust_aggregation(A, B)

    # Print the result (with the .numpy() typo corrected)
    print("Result of A @ B:")
    print(tf.sparse.to_dense(C).numpy())
