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

def compute_sparse_C(sparse_C_computation_coordinates, dense_A, dense_B, dense_result=False):
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
