import numpy as np


def pqp(Q, h, x=None, iters=None, maxval=None):
    """
    Parallel Quadratic Programming (PQP) multiplicative update.

    Minimizes
        f(x) = 0.5 * x^T Q x - h^T x
    subject to
        x >= 0

    Parameters
    ----------
    Q : (N, N) array_like
        Symmetric positive semidefinite matrix.
    h : (N,) array_like
        Vector.
    x : (N,) array_like, optional
        Initial guess for x. If None, a heuristic guess is used.
    iters : int, optional
        Maximum number of iterations. Default: 10000.
    maxval : float or (N,) array_like, optional
        Optional upper bound(s) for x (box constraint).

    Returns
    -------
    x : (N,) ndarray
        Final iterate of PQP multiplicative update.
    values : (k,) ndarray
        History of objective values f(x) over iterations (including initial).
    """
    # Convert inputs to numpy arrays
    Q = np.asarray(Q, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1)

    # INPUT CHECKING
    if h.ndim != 1:
        raise ValueError("h must be a 1D vector.")

    N = h.size
    if Q.shape != (N, N):
        raise ValueError("Q and h are inconsistent sizes: "
                         f"Q.shape={Q.shape}, len(h)={N}")

    # ALGORITHM PARAMETERS
    epsilon = 1e-6                 # avoid 1/0 errors
    threshold = N * epsilon        # convergence threshold

    # SPLIT
    Qp = np.maximum(Q, 0.0)        # simplest possible split
    hp = np.maximum(h, 0.0)

    # ...CONTRACTION GUARANTEE: modify the split for semidefinite problems
    # (the two optional lines from MATLAB are left commented here as well)
    # Qp[np.arange(N), np.arange(N)] += np.sum(Qp - Q)     # via diagonal dominance
    # Qp = np.abs(Q)                                      # depends on Q and x

    # Improve conditioning
    diag_idx = np.diag_indices(N)
    Qp[diag_idx] = Qp[diag_idx] + epsilon
    hp = hp * (1.0 + epsilon)

    # Complete split
    Qn = Qp - Q
    hn = hp - h

    # INITIALIZATION
    if x is None or (isinstance(x, np.ndarray) and x.size == 0):
        diag_Q = np.diag(Q)
        # same heuristic as MATLAB: (mean(abs(h)) + abs(h)) ./ diag(Q)
        x = (np.mean(np.abs(h)) + np.abs(h)) / diag_Q
    else:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != N:
            raise ValueError("Initial x must have length N = len(h).")

    if iters is None:
        iters = int(1e4)

    # Handle optional maxval
    if maxval is not None:
        maxval = np.asarray(maxval, dtype=float)
        # Allow scalar or vector upper bounds
        # Broadcasting will handle shapes like (N,) or scalar.

    # Objective history
    values = []
    f_val = 0.5 * x @ (Q @ x) - h @ x
    values.append(f_val)

    # ALGORITHM
    for i in range(1, iters + 1):
        # OPTIONAL: bound x away from zero to avoid boundary traps
        x = np.maximum(x, epsilon)

        # PQP fixpoint update: x = x .* (Qn*x + hp) ./ (Qp*x + hn)
        num = Qn @ x + hp
        den = Qp @ x + hn
        x = x * (num / den)

        # OPTIONAL: force upper bound
        if maxval is not None:
            x = np.minimum(x, maxval)

        # Record objective
        f_val = 0.5 * x @ (Q @ x) - h @ x
        values.append(f_val)

        # KKT convergence check every 32 iterations
        if i % 32 == 0:
            kkt = np.max(x * np.abs(Q @ x - h))
            if kkt < threshold:
                break

    return x, np.array(values)

class PQPModel:
    """
    Light wrapper around the PQP solver to mimic an OSQP-like interface:

        model = PQPModel()
        model.setup(Q0, h0, x0, iters=..., maxval=...)
        ...
        model.update(Q=Qk, h=hk)   # per MPC step
        res = model.solve()        # uses warm start

    It doesn't do factorization like OSQP; it just keeps problem data and
    a warm-started x between solves.
    """

    def __init__(self):
        self.Q = None
        self.h = None
        self.x = None
        self.iters = None
        self.maxval = None
        self.n = None

    def setup(self, Q, h, x0=None, iters=10_000, maxval=None):
        """
        One-time initial setup. You can call this again if you really
        want to change dimension, but the MPC-like pattern is:
          - call once,
          - then only call update() and solve().
        """
        Q = np.asarray(Q, dtype=float)
        h = np.asarray(h, dtype=float).reshape(-1)

        if h.ndim != 1:
            raise ValueError("h must be a 1D vector.")
        if Q.shape != (h.size, h.size):
            raise ValueError(f"Inconsistent sizes: Q.shape={Q.shape}, h.size={h.size}")

        self.n = h.size
        self.Q = Q.copy()
        self.h = h.copy()
        self.iters = int(iters)
        self.maxval = None if maxval is None else np.asarray(maxval, dtype=float)

        if x0 is None:
            # Let pqp() pick its own heuristic initial guess on first call.
            self.x = None
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
            if x0.size != self.n:
                raise ValueError("x0 must have same length as h.")
            self.x = x0.copy()

    def update(self, Q=None, h=None, maxval=None, iters=None):
        """
        Update problem data before calling solve(), similar to OSQP.update().
        Only changes what you pass in.
        """
        if Q is not None:
            Q = np.asarray(Q, dtype=float)
            if Q.shape != (self.n, self.n):
                raise ValueError(f"New Q has wrong shape: {Q.shape}, expected {(self.n, self.n)}")
            self.Q = Q

        if h is not None:
            h = np.asarray(h, dtype=float).reshape(-1)
            if h.size != self.n:
                raise ValueError(f"New h has wrong length: {h.size}, expected {self.n}")
            self.h = h

        if maxval is not None:
            self.maxval = np.asarray(maxval, dtype=float)

        if iters is not None:
            self.iters = int(iters)

