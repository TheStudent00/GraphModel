import tensorflow as tf

"""
  generalized sparse-to-sparse multiplication leveraging TF computational graphs and GPU optimizations
"""


# Define an input signature to create a single, reusable graph
SPARSE_MUL_SIGNATURE = [
    tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
    tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32)
]

@tf.function(input_signature=SPARSE_MUL_SIGNATURE)
def sparse_multiplication(A_sparse: tf.SparseTensor, B_sparse: tf.SparseTensor) -> tf.SparseTensor:
    """
    performs Sparse-Sparse-Multiplication using a method that explicitly
    sums values for duplicate indices before final tensor construction.

    it only works for 2D sparse matrices.

    sparse matrices are used when matrices have roughly a third non-zero elements.

    the current implementation does not use an algorithmic coordination.
    it relies on fully expressing all the dot products.
    doing this coordination is possible, 
    however, it is challenging to do so within Python library APIs.
    that being said, the current implementation is not far off.

    for example:
      A:
      [[a11, a12, a13],
       [a21, a22, a23],
       [a31, a32, a33]]
      
      B:
      [[b11, b12, b13],
       [b21, b22, b23],
       [b31, b32, b33]]
      
      C:
      [[a11*b11+a12*b21+a13*b31, a11*b12+a12*b22+a13*b32, a11*b13+a12*b23+a13*b33],
       [a21*b11+a22*b21+a23*b31, a21*b12+a22*b22+a23*b32, a21*b13+a22*b23+a23*b33],
       [a31*b11+a32*b21+a33*b31, a31*b12+a32*b22+a33*b32, a31*b13+a32*b23+a33*b33]]

    the above matrices are symbolic. some values might be zero. 
    observing the element of Matrix C at C[0,0]:
        a11*b11+a12*b21+a13*b31
    
    each element of Matrix C has the same kind of pattern:
        a_ik*b_kj + ...
    the kth columnn index from Matrix A always matches the kth row index from Matrix B.
    the ith row index from Matrix A and jth from Matrix B create the location in Matrix C.

    a sparse tensor contains all these indices in a nice form for such matching,
    and is leveraged in this implementation.

    """
    # find all products and their destination coordinates
    # adjusting the structure for matching indexes
    A_indices, A_values = A_sparse.indices, A_sparse.values
    B_indices, B_values = B_sparse.indices, B_sparse.values
    A_i, A_k = A_indices[:, 0], A_indices[:, 1]
    B_k, B_j = B_indices[:, 0], B_indices[:, 1]

    # matching the column indexes of Matrix A (A_sparse) with the row indexes of Matrix B (B_sparse) 
    matching_k = tf.equal(A_k[:, None], B_k[None, :])
    match_indices = tf.where(matching_k)
    A_match_index, B_match_index = match_indices[:, 0], match_indices[:, 1]

    # gather the numerical elements corresponding to matched indexes
    all_C_i = tf.gather(A_i, A_match_index)
    all_C_j = tf.gather(B_j, B_match_index)
    all_C_values = tf.gather(A_values, A_match_index) * tf.gather(B_values, B_match_index)

    # generate segment ids for coordinating the summation of products with matching indexes
    output_num_cols = tf.cast(B_sparse.dense_shape[1], tf.int64)
    segment_ids_flat = all_C_i * output_num_cols + all_C_j
    
    # use tf.unique to find the final unique output coordinates and the mapping.
    unique_flat_indices, segment_mapping = tf.unique(segment_ids_flat)

    # use segment_sum to sum the values that belong to the same coordinate.
    summed_values = tf.math.segment_sum(all_C_values, segment_mapping)

    # 5. Assemble the final SparseTensor from the unique indices and summed values.
    # Convert the unique 1D IDs back to [row, col] format.
    final_C_i = unique_flat_indices // output_num_cols
    final_C_j = unique_flat_indices % output_num_cols
    final_C_indices = tf.stack([final_C_i, final_C_j], axis=1)
    
    output_shape = (A_sparse.dense_shape[0], B_sparse.dense_shape[1])
    
    return tf.SparseTensor(
        indices=final_C_indices,
        values=summed_values,
        dense_shape=tf.cast(output_shape, tf.int64)
    )

if __name__ == "__main__":
    # example
    A = tf.sparse.from_dense(tf.constant([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]], dtype=tf.float32))
    B = tf.sparse.from_dense(tf.constant([[5.0, 6.0], [7.0, 0.0], [0.0, 8.0]], dtype=tf.float32))
    C = sparse_multiplication(A, B)

    print("Result of A @ B:")
    print(tf.sparse.to_dense(C).numpy())
