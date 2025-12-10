import numpy as NP
import casadi as C

def qpsolve(
    H,
    g,
    lbx,
    ubx,
    A=NP.zeros((0, 0)),
    lba=NP.zeros(0),
    uba=NP.zeros(0),
    termination_tol=1e-6,
    verbose=False,
    x0=None,        # <-- new: primal warm start
    lam_x0=None,    # <-- optional: dual warm start for simple bounds
    lam_a0=None,    # <-- optional: dual warm start for linear constraints
):
    # Convert to CasADi DM
    H   = C.DM(H)
    g   = C.DM(g).reshape((-1, 1))
    lbx = C.DM(lbx).reshape((-1, 1))
    ubx = C.DM(ubx).reshape((-1, 1))
    A   = C.DM(A)
    A   = A.reshape((A.size1(), H.size1()))  # (nc, nx)
    lba = C.DM(lba).reshape((-1, 1))
    uba = C.DM(uba).reshape((-1, 1))

    # Optional warm-starts
    if x0 is not None:
        x0 = C.DM(x0).reshape((-1, 1))
    if lam_x0 is not None:
        lam_x0 = C.DM(lam_x0).reshape((-1, 1))
    if lam_a0 is not None:
        lam_a0 = C.DM(lam_a0).reshape((-1, 1))

    # Structure (sparsity only) for low-level QP interface
    qp_struct = {
        "h": H.sparsity(),
        "a": A.sparsity(),
    }

    opts = {
        "terminationTolerance": float(termination_tol),
        "verbose": False,
        "print_time": False,
        "printLevel": "none" if not verbose else "medium",
    }

    # Use the *conic* interface with the qpoases plugin
    solver = C.conic("solver", "qpoases", qp_struct, opts)

    # Build input dict for the solver
    inputs = dict(
        h=H,
        g=g,
        a=A,
        lbx=lbx,
        ubx=ubx,
        lba=lba,
        uba=uba,
    )

    # Attach warm-start inputs if provided
    if x0 is not None:
        inputs["x0"] = x0
    if lam_x0 is not None:
        inputs["lam_x0"] = lam_x0
    if lam_a0 is not None:
        inputs["lam_a0"] = lam_a0

    # Call the solver with numeric data
    sol = solver(**inputs)

    x_opt = NP.array(sol["x"]).reshape((-1,))
    return x_opt
