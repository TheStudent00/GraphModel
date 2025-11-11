"""
  about as optimized as generalized sparse-sparse-multiplication can be within the TensorFlow/GPU API
"""

import tensorflow as tf

SPARSE_MUL_SIGNATURE = [
    tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
    tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32)
]

@tf.function(input_signature=SPARSE_MUL_SIGNATURE)
def sparse_mul(A_sp: tf.SparseTensor, B_sp: tf.SparseTensor) -> tf.SparseTensor:
    """
    Implements SpGEMM by first determining all contributing coordinates,
    then computing values, and finally aggregating.
    """
    # --- Deconstruct Inputs (Integer Indices Only) ---
    A_i, A_k = A_sp.indices[:, 0], A_sp.indices[:, 1]
    B_k, B_j = B_sp.indices[:, 0], B_sp.indices[:, 1]
    
    # --- PHASE 1: Construct C_ij Coordinates using Integer Indexes ---
    # Find all connections between A's columns and B's rows on the inner dimension `k`.
    # This is the 'join' that identifies every single product that needs to be computed.
    matching_k = tf.equal(A_k[:, None], B_k[None, :])
    match_indices = tf.where(matching_k)
    A_match_idx, B_match_idx = match_indices[:, 0], match_indices[:, 1]

    # Use the join results to construct the flat list of all C_ij coordinates
    # that will receive a non-zero contribution. This list will contain duplicates.
    all_C_i = tf.gather(A_i, A_match_idx)
    all_C_j = tf.gather(B_j, B_match_idx)

    # --- PHASE 2: Compute Values for the Found Coordinates ---
    # Now that we know which products to compute, gather the float values.
    A_vals_matched = tf.gather(A_sp.values, A_match_idx)
    B_vals_matched = tf.gather(B_sp.values, B_match_idx)
    
    # Perform all floating-point multiplications.
    all_C_values = A_vals_matched * B_vals_matched

    # --- PHASE 3: Aggregate Results into Final Sparse Output ---
    # Create unique integer IDs for each (i, j) coordinate pair to prepare for summation.
    output_num_cols = tf.cast(B_sp.dense_shape[1], tf.int64)
    segment_ids_flat = all_C_i * output_num_cols + all_C_j
    
    unique_flat_indices, segment_mapping = tf.unique(segment_ids_flat)

    # Sum the product values that belong to the same final C_ij coordinate.
    summed_values = tf.math.segment_sum(all_C_values, segment_mapping)

    # Assemble the final SparseTensor from the unique coordinates and summed values.
    final_C_i = unique_flat_indices // output_num_cols
    final_C_j = unique_flat_indices % output_num_cols
    final_C_indices = tf.stack([final_C_i, final_C_j], axis=1)
    
    output_shape = (A_sp.dense_shape[0], B_sp.dense_shape[1])
    
    return tf.SparseTensor(
        indices=final_C_indices,
        values=summed_values,
        dense_shape=tf.cast(output_shape, tf.int64)
    )

# --- Example Data ---
if __name__ == "__main__":
    sparse_indices_A = tf.constant([[0,1], [0,4], [0,7], [2,2], [2,3], [2,8], [4,1], [8,8]], dtype=tf.int64)
    sparse_values_A = tf.range(1, 1 + tf.shape(sparse_indices_A)[0], dtype=tf.float32)
    dense_shape_A   = tf.constant([9, 9], dtype=tf.int64)
    sparse_A = tf.SparseTensor(sparse_indices_A, sparse_values_A, dense_shape_A)

    sparse_indices_B = tf.constant([[0,1], [0,4], [0,7], [2,3], [2,8], [4,0], [4,3], [4,8], [7,7]], dtype=tf.int64)
    sparse_values_B = tf.range(1, 1 + tf.shape(sparse_indices_B)[0], dtype=tf.float32)
    dense_shape_B   = tf.constant([9, 9], dtype=tf.int64)
    sparse_B = tf.SparseTensor(sparse_indices_B, sparse_values_B, dense_shape_B)

    # --- Run Computation ---
    sparse_C = sparse_mul(sparse_A, sparse_B)
    print(sparse_C)
    dense_C_reference = tf.sparse.to_dense(sparse_A) @ tf.sparse.to_dense(sparse_B)
    print(dense_C_reference)
