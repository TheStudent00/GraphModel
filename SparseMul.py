"""
  about as optimized as generalized sparse-sparse-multiplication can be within the TensorFlow library
    perhaps any Python library with GPU-API
"""

import tensorflow as tf
import sys

# Helper function for printing
def kw_print(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}:\n {value.numpy() if hasattr(value, 'numpy') else value}\n")
    sys.stdout.flush()

# @tf.function
def extract_shared(A_tensor: tf.Tensor, B_tensor: tf.Tensor):
    """Finds the intersection of two 1D tensors."""
    shape_A = tf.cast(tf.shape(A_tensor), tf.int64)
    sparse_set_A = tf.SparseTensor(
        indices=tf.stack([tf.zeros(shape_A, dtype=tf.int64),
                        tf.range(shape_A[0], dtype=tf.int64)], axis=1),
        values=A_tensor,
        dense_shape=[1, shape_A[0]]
    )
    shape_B = tf.cast(tf.shape(B_tensor), tf.int64)
    sparse_set_B = tf.SparseTensor(
        indices=tf.stack([tf.zeros(shape_B, dtype=tf.int64),
                        tf.range(shape_B[0], dtype=tf.int64)], axis=1),
        values=B_tensor,
        dense_shape=[1, shape_B[0]]
    )
    return tf.sets.intersection(sparse_set_A, sparse_set_B).values

# @tf.function
def sparse_vector_filter_by_k_indices(
        A_sparse: tf.SparseTensor, 
        index_i: tf.Tensor, 
        shared_k_indices: tf.Tensor) -> tf.Tensor:
    """
    For a given row `index_i`, retrieves values at column locations `shared_k_indices`.
    Returns 0.0 for non-existent values.
    """
    # Isolate all non-zero elements in the target row 'index_i'.
    row_mask = tf.equal(A_sparse.indices[:, 0], index_i)
    row_indices = tf.boolean_mask(A_sparse.indices, row_mask)
    row_values = tf.boolean_mask(A_sparse.values, row_mask)
    row_ks = row_indices[:, 1]
    
    # Check if the row has any non-zero elements.
    num_elements_in_row = tf.shape(row_ks)[0]

    # Use tf.cond to handle the empty case gracefully.
    return tf.cond(
        tf.greater(num_elements_in_row, 0),
        # If the row is not empty, build the hash table and lookup.
        lambda: tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys=row_ks, values=row_values),
            default_value=0.0
        ).lookup(shared_k_indices),
        # Otherwise, if the row is empty, just return a tensor of zeros.
        lambda: tf.zeros_like(shared_k_indices, dtype=A_sparse.values.dtype)
    )

# @tf.function
def product_sum(A_sparse, B_sparse, index_ij, shared_k_indices):
    """Calculates the dot product for a single output element C[i, j]."""
    # For C[i,j], we need row i from A and row j from B (transposed)
    # So we look for B's values in column j.
    # We need to filter B's columns, not rows. Let's adjust B.
    B_transposed = tf.sparse.transpose(B_sparse)

    # Get row i from A and row j from B_transposed (which is col j of B)
    A_row_i_values = sparse_vector_filter_by_k_indices(A_sparse, index_ij[0], shared_k_indices)
    B_col_j_values = sparse_vector_filter_by_k_indices(B_transposed, index_ij[1], shared_k_indices)
    
    return tf.reduce_sum(A_row_i_values * B_col_j_values)

# @tf.function
def compute_sparse_mul(A_sparse: tf.SparseTensor, B_sparse: tf.SparseTensor):
    """Computes sparse-sparse matrix multiplication."""
    A_indices, _ = A_sparse.indices, A_sparse.values
    B_indices, _ = B_sparse.indices, B_sparse.values
    A_i, A_k = A_indices[:, 0], A_indices[:, 1]
    B_k, B_j = B_indices[:, 0], B_indices[:, 1]

    # Find the k indices that are relevant for multiplication
    A_k_unique, _ = tf.unique(A_k)
    B_k_unique, _ = tf.unique(B_k)
    shared_k_indices = extract_shared(A_k_unique, B_k_unique)

    # Find all potential output row indices `i` from A
    A_i_mask = tf.reduce_any(tf.equal(A_k[:, None], shared_k_indices[None, :]), axis=1)
    filtered_A_i = tf.boolean_mask(A_i, A_i_mask)
    
    # Find all potential output column indices `j` from B
    B_k_mask = tf.reduce_any(tf.equal(B_k[:, None], shared_k_indices[None, :]), axis=1)
    filtered_B_j = tf.boolean_mask(B_j, B_k_mask)
    
    unique_filtered_A_i, _ = tf.unique(filtered_A_i)
    unique_filtered_B_j, _ = tf.unique(filtered_B_j)

    # Create all (i, j) pairs to compute
    A_i_mesh, B_j_mesh = tf.meshgrid(unique_filtered_A_i, unique_filtered_B_j)
    C_indices_to_compute = tf.stack([tf.reshape(A_i_mesh, [-1]), tf.reshape(B_j_mesh, [-1])], axis=1)
    
    # Use tf.map_fn to run the product_sum for each (i,j) pair in parallel
    C_values = tf.map_fn(
        lambda index_ij: product_sum(A_sparse, B_sparse, index_ij, shared_k_indices),
        C_indices_to_compute,
        fn_output_signature=tf.float32
    )

    # Filter out any results that are zero
    non_zero_mask = tf.not_equal(C_values, 0.0)
    final_indices = tf.boolean_mask(C_indices_to_compute, non_zero_mask)
    final_values = tf.boolean_mask(C_values, non_zero_mask)
    
    output_shape = (A_sparse.dense_shape[0], B_sparse.dense_shape[1])
    
    # Construct the final, but potentially unordered, sparse tensor
    unordered_sparse_C = tf.SparseTensor(indices=final_indices, values=final_values, dense_shape=output_shape)
    
    # Reorder the tensor to ensure indices are sorted, as required by many TF sparse ops.
    return tf.sparse.reorder(unordered_sparse_C)

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
    sparse_C = compute_sparse_mul(sparse_A, sparse_B)
    dense_C_reference = tf.sparse.to_dense(sparse_A) @ tf.sparse.to_dense(sparse_B)

    print("--- Custom Sparse Multiplication Result ---")
    kw_print(sparse_C_dense=tf.sparse.to_dense(sparse_C))

    print("\n--- TF Dense Multiplication Result (for verification) ---")
    kw_print(dense_C=dense_C_reference)
