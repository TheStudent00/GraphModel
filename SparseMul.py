""" 
  rough draft without vectorization such as: 
    * tf.vectorized_map 
    * limited control flow compatible fixed computational graph
"""

def get_sparse_row_col(sparse_A):
  sparse_A_row = {}
  sparse_A_col = {}
  for i in range(len(sparse_A)):
    row, col = sparse_A[i]
    if row not in sparse_A_row:
      sparse_A_row[row] = set()
    sparse_A_row[row].add(col)
    if col not in sparse_A_col:
      sparse_A_col[col] = set()
    sparse_A_col[col].add(row)
  return sparse_A_row, sparse_A_col

def get_sparse_product_map(sparse_A_cols_dict, sparse_B_rows_dict):
  """
    A: 
      [[a0, a1, a2],
       [b0, b1, b2],
       [c0, c1, c2]]

    sparse_A_cols_dict:
      {
        0: {a, b, c},
        1: {a, b, c},
        2: {a, b, c}
      }


    B: 
      [[0a, 0b, 0c],
       [1a, 1b, 1c],
       [2a, 2b, 2c]]

    sparse_B_rows_dict:
      {
        a: {0, 1, 2},
        b: {0, 1, 2},
        c: {0, 1, 2}
      }

    sparse_C_computation_coordinates:
      {
        (a, a): {(0, 0), (1, 0), (2, 0)},
        (a, b): {(0, 1), (1, 1), (2, 1)},
        (a, c): {(0, 2), (1, 2), (2, 2)},
        (b, a): {(0, 0), (1, 0), (2, 0)},
        (b, b): {(0, 1), (1, 1), (2, 1)},
        (b, c): {(0, 2), (1, 2), (2, 2)},
        (c, a): {(0, 0), (1, 0), (2, 0)},
        (c, b): {(0, 1), (1, 1), (2, 1)},
        (c, c): {(0, 2), (1, 2), (2, 2)}
      }
  """
  sparse_C_computation_coordinates = {}

  for sparse_A_col_key in sparse_A_cols_dict:
    if sparse_A_col_key in sparse_B_rows_dict:
      for sparse_A_col_value in sparse_A_cols_dict[sparse_A_col_key]:
        for sparse_B_row_value in sparse_B_rows_dict[sparse_A_col_key]:
          if (sparse_A_col_value, sparse_B_row_value) not in sparse_C_computation_coordinates:
            sparse_C_computation_coordinates[(sparse_A_col_value, sparse_B_row_value)] = set()
          new_pair = ((sparse_A_col_value, sparse_A_col_key), (sparse_A_col_key, sparse_B_row_value))
          sparse_C_computation_coordinates[(sparse_A_col_value, sparse_B_row_value)].add(new_pair)
  return sparse_C_computation_coordinates

def compute_sparse_C_with_map(sparse_C_computation_coordinates, dense_A, dense_B, dense_result=False):
  if dense_result:
    dense_C = np.zeros_like(dense_A)
    for sparse_C_key in sparse_C_computation_coordinates:
      for sparse_C_value in sparse_C_computation_coordinates[sparse_C_key]:
        dense_A_elem = dense_A[sparse_C_value[0][0], sparse_C_value[0][1]]
        dense_B_elem = dense_B[sparse_C_value[1][0], sparse_C_value[1][1]]
        dense_C[sparse_C_key[0], sparse_C_key[1]] += dense_A[sparse_C_value[0][0], sparse_C_value[0][1]] * dense_B[sparse_C_value[1][0], sparse_C_value[1][1]]
  
  sparse_C = {}
  for sparse_C_key in sparse_C_computation_coordinates:
    sparse_C[sparse_C_key] = 0
    for sparse_C_value in sparse_C_computation_coordinates[sparse_C_key]:
      dense_A_elem = dense_A[sparse_C_value[0][0], sparse_C_value[0][1]]
      dense_B_elem = dense_B[sparse_C_value[1][0], sparse_C_value[1][1]]
      sparse_C[sparse_C_key] += dense_A_elem * dense_B_elem

  if dense_result:
    return sparse_C, dense_C
  else:
    return sparse_C

def compute_sparse_C(sparse_A, sparse_B, sparse_A_cols_dict, sparse_B_rows_dict):
  sparse_C = {}
  for sparse_A_col_key in sparse_A_cols_dict:
    if sparse_A_col_key in sparse_B_rows_dict:
      for sparse_A_col_value in sparse_A_cols_dict[sparse_A_col_key]:
        for sparse_B_row_value in sparse_B_rows_dict[sparse_A_col_key]:
          if (sparse_A_col_value, sparse_B_row_value) not in sparse_C:
            sparse_C[(sparse_A_col_value, sparse_B_row_value)] = 0
          sparse_A_elem = sparse_A[(sparse_A_col_value, sparse_A_col_key)]
          sparse_B_elem = sparse_B[(sparse_A_col_key, sparse_B_row_value)]
          summand = sparse_A_elem * sparse_B_elem
          sparse_C[(sparse_A_col_value, sparse_B_row_value)] += summand
  return sparse_C


"""
Gemini's respectable effort to implement vectorization of above algorithm:
import tensorflow as tf
import numpy as np

def vectorized_compute_sparse_c(A_sp: tf.SparseTensor, B_sp: tf.SparseTensor) -> tf.SparseTensor:
    """
    A vectorized TensorFlow implementation of the on-the-fly sparse
    matrix multiplication algorithm.
    """
    # 1. Deconstruct the sparse tensors into their essential components.
    # This is equivalent to your pre-processed dictionaries.
    A_indices, A_values = A_sp.indices, A_sp.values
    B_indices, B_values = B_sp.indices, B_sp.values

    # A_i represents row indices from A, A_k is the shared inner dimension.
    A_i, A_k = A_indices[:, 0], A_indices[:, 1]
    # B_k is the shared inner dimension, B_j represents column indices from B.
    B_k, B_j = B_indices[:, 0], B_indices[:, 1]

    # 2. Find all valid (A_ik, B_kj) pairs where the inner dimension `k` matches.
    # This single broadcasted `tf.equal` operation replaces your outer loop:
    # `for sparse_A_col_key in sparse_A_cols_dict: if sparse_A_col_key in sparse_B_rows_dict:`
    matching_k = tf.equal(A_k[:, None], B_k[None, :])
    
    # Get the indices [index_in_A, index_in_B] for every valid product.
    match_indices = tf.where(matching_k)
    A_match_idx, B_match_idx = match_indices[:, 0], match_indices[:, 1]

    # 3. Gather all components needed for the multiplication.
    # These `gather` operations effectively perform the work of your inner loops:
    # `for sparse_A_col_value...` and `for sparse_B_row_value...`
    C_i = tf.gather(A_i, A_match_idx)
    C_j = tf.gather(B_j, B_match_idx)

    A_vals_matched = tf.gather(A_values, A_match_idx)
    B_vals_matched = tf.gather(B_values, B_match_idx)
    
    # 4. Perform all multiplications in one vectorized operation.
    # This corresponds to `sparse_A_elem * sparse_B_elem`.
    C_vals_pre_sum = A_vals_matched * B_vals_matched

    # 5. Sum all products that contribute to the same output coordinate C[i, j].
    # This vectorized operation replaces the `sparse_C[...] += summand` logic.
    
    # Create a unique integer ID for each (i, j) coordinate.
    output_num_cols = tf.cast(B_sp.dense_shape[1], tf.int64)
    segment_ids = tf.cast(C_i, tf.int64) * output_num_cols + tf.cast(C_j, tf.int64)

    # Sum the values based on their segment ID.
    summed_C_values = tf.math.segment_sum(C_vals_pre_sum, segment_ids)

    # Get the unique (i, j) coordinates corresponding to the summed values.
    unique_C_indices_flat = tf.unique(segment_ids)[0]
    final_C_i = unique_C_indices_flat // output_num_cols
    final_C_j = unique_C_indices_flat % output_num_cols
    final_C_indices = tf.stack([final_C_i, final_C_j], axis=1)

    # 6. Reconstruct the final sparse tensor from the results.
    output_shape = (A_sp.dense_shape[0], B_sp.dense_shape[1])
    C_sp = tf.SparseTensor(
        indices=final_C_indices,
        values=summed_C_values,
        dense_shape=tf.cast(output_shape, tf.int64)
    )
    
    return tf.sparse.reorder(C_sp)
"""
