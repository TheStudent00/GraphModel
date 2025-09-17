class SparseMaskedLayer:
    def __init__(self, in_dim, out_dim, mask_init, l1=0.0):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.K = Parameter(shape=(in_dim, out_dim))      # dense trainable
        self.M = mask_init.astype(bool)                  # non-trainable
        self.b = Parameter(shape=(out_dim,))
        self.l1 = l1

    def forward(self, X):
        K_eff = self.K * self.M                          # elementwise
        return X @ K_eff + self.b

    def step_post_update(self):
        # proximal L1 on allowed entries only
        if self.l1 > 0:
            K = self.K
            K[self.M] = sign(K[self.M]) * relu(abs(K[self.M]) - self.l1)
            self.K = K

    def complexity(self):
        # choose: structural budget or realized nnz
        return int(self.M.sum())        # structural
        # or: (abs(self.K) > eps & self.M).sum()  # realized
