import osqp
import numpy as np
import scipy.sparse as sp
from genqp import skew, Ib
from uprightmpc2py import UprightMPC2C # C version
import time
from pqp import *
from qpsolvers import solve_qp
# from qpsolvers import solve_qp
from casadiQPOases import qpsolve as casadi_qpsolve

"""This file creates the python controller, 
and can return py ver, C ver, and a reactive"""

ny = 6
nu = 3

# Basic constituents of dynamics A0, B0 (only sparsity matters)
def getA0(T0):
    A0 = np.zeros((6, 6))
    A0[:3,3:] = T0*np.eye(3)
    return A0

def getB0(s0, Btau):
    return np.block([
        [np.reshape(s0, (3,1)),np.zeros((3,2))],
        [np.zeros((3,1)), Btau]
    ])
    
c0 = lambda g : np.array([0,0,-g,0,0,0])

e3h = skew([0,0,1])

def initConstraint(N, nx, nc):
    # these will be updated
    T0 = 1
    dt = 1
    s0 = np.ones(3)
    Btau = np.ones((3,2))

    A = np.zeros((nc, nx))
    P = np.eye(nx) # Q, R stacked

    # cols of A are broken up like this (partition nx)
    n1 = N*ny
    n2 = 2*N*ny
    # rows of A broken up:
    nc1 = N*ny # after this; yddot dynamics
    nc2 = 2*N*ny # after this; thrust lims
    
    for k in range(N):
        # ykp1 = yk + dt*dyk equations
        A[k*ny:(k+1)*ny, k*ny:(k+1)*ny] = -np.eye(ny)
        A[k*ny:(k+1)*ny, n1 + k*ny:n1 + (k+1)*ny] = dt * np.eye(ny)
        if k>0:
            A[k*ny:(k+1)*ny, (k-1)*ny:(k)*ny] = np.eye(ny)
        
        # dykp1 equation
        A[nc1 + k*ny:nc1 + (k+1)*ny, n1 + k*ny:n1 + (k+1)*ny] = -np.eye(ny)
        A[nc1 + k*ny:nc1 + (k+1)*ny, n2 + k*nu:n2 + (k+1)*nu] = getB0(s0, Btau)
        if k>0:
            A[nc1 + k*ny:nc1 + (k+1)*ny, n1 + (k-1)*ny:n1 + (k)*ny] = np.eye(ny)
        if k>1:
            A[nc1 + k*ny:nc1 + (k+1)*ny, (k-2)*ny:(k-1)*ny] = getA0(T0*dt)
        
        # thrust lim
        A[nc2+k, n2+3*k] = 1
    
    return sp.csc_matrix(A), sp.csc_matrix(P)

def updateConstraint(N, A, dt, T0, s0s, Btaus, y0, dy0, g, Tmax):
    nc = A.shape[0]

    # Update vector
    l = np.zeros(nc)
    y1 = y0 + dt * dy0
    l[:ny] = -y1
    for k in range(N):
        if k == 0:
            l[ny*N+k*ny : ny*N+(k+1)*ny] = -dy0 - dt*getA0(T0) @ y0 - dt*c0(g)
        elif k == 1:
            l[ny*N+k*ny : ny*N+(k+1)*ny] = -dt*getA0(T0) @ y1 - dt*c0(g)
        else:
            l[ny*N+k*ny : ny*N+(k+1)*ny] = -dt*c0(g)
    # copy for dynamics
    u = np.copy(l)
    # thrust lims
    for k in range(N):
        l[2*N*ny+k] = -T0
        u[2*N*ny+k] = Tmax-T0

    # Left third
    AxidxT0dt = []
    n2 = 2*ny + 3 # nnz in each block col on the left
    for k in range(N-2):
        AxidxT0dt += [n2*k + i for i in [8,11,14]]

    # Middle third
    n1 = (2*N-1)*ny + (N-2)*3 # All the nnz in the left third
    n2 = 3*ny # nnz in each of the first N-1 block cols in the middle third
    Axidxdt = []
    for k in range(N):
        if k < N-1:
            Axidxdt += [n1 + n2*k + i for i in [0,3,6,9,12,15]]
        else:
            Axidxdt += [n1 + n2*k + i for i in [0,2,4,6,8,10]]
    
    # Right third
    n1 += 3*ny*(N-1) + 2*ny # all nnz in the left and middle third
    n2 = 10 # nnz in each B0 + 1 for thrust lim
    Axidxs0 = []
    AxidxBtau = []
    for k in range(N):
        Axidxs0 += [n1 + n2*k + i for i in range(3)]
        AxidxBtau += [n1 + n2*k + 4 + i for i in range(6)]
    # No need to update rightmost

    # Last check
    assert A.nnz == n1 + n2*N

    # Update
    A.data[AxidxT0dt] = dt*T0
    A.data[Axidxdt] = dt
    A.data[Axidxs0] = dt*np.hstack((s0s))
    A.data[AxidxBtau] = dt*np.hstack([np.ravel(Btau,order='F') for Btau in Btaus])

    Axidx = np.hstack((AxidxT0dt, Axidxdt, Axidxs0, AxidxBtau))
    # print("nAdata =",len(Axidx))

    # print(A[:,2*N*ny:2*N*ny+6].toarray())
    fullA = A.toarray()
    return A, l, u, Axidx

def updateObjective(N, Qyr, Qyf, Qdyr, Qdyf, R, ydes, dydes):
    # Block diag components - see notes
    Pdata = np.hstack((
        np.hstack([Qyr for k in range(N-1)]),
        Qyf,
        np.hstack([Qdyr for k in range(N-1)]),
        Qdyf,
        np.hstack([R for k in range(N)])
    ))
    q = np.hstack((
        np.hstack([-Qyr*ydes for k in range(N-1)]),
        -Qyf*ydes,
        np.hstack([-Qdyr*dydes for k in range(N-1)]),
        -Qdyf*dydes,
        np.zeros(N*len(R))
    ))
    return Pdata, q

def openLoopX(N, dt, T0, s0s, Btaus, y0, dy0, g):
    ys = np.zeros((N,ny))
    dys = np.zeros((N,ny))
    us = np.random.rand(N,nu)

    yk = np.copy(y0)
    dyk = np.copy(dy0)
    for k in range(N):
        # at k=0, yy=y0, 
        dykp1 = dyk + dt*(getA0(T0) @ yk + getB0(s0s[k], Btaus[k]) @ us[k,:] + c0(g)) # dy[k+1]

        dys[k,:] = dykp1
        ykp1 = yk + dt * dyk # y[k+1]
        ys[k,:] = ykp1 + dt * dykp1 # y[k+2]

        # For next k
        yk = np.copy(ykp1)
        dyk = np.copy(dykp1)

    # stack
    x = np.hstack((np.ravel(ys), np.ravel(dys), np.ravel(us)))
    # print(ys, x)
    return x

class UprightMPC2():
    def __init__(self, N, dt, g, TtoWmax, ws, wds, wpr, wpf, wvr, wvf, wthrust, wmom, Ib,eps_rel_in,eps_abs_in):
        self.N = N

        nx = self.N * (2*ny + nu)
        nc = 2*self.N*ny + self.N

        self.A, self.P = initConstraint(N, nx, nc)

        Qyr = np.hstack((np.full(3,wpr), np.full(3,ws)))
        Qyf = np.hstack((np.full(3,wpf), np.full(3,ws)))
        Qdyr = np.hstack((np.full(3,wvr), np.full(3,wds)))
        Qdyf = np.hstack((np.full(3,wvf), np.full(3,wds)))
        R = np.hstack((wthrust,np.full(2,wmom)))

        self.dt = dt
        self.Wts = [Qyr, Qyf, Qdyr, Qdyf, R]
        self.g = g
        self.Tmax = TtoWmax * g # use thrust-to-weight ratio to set max specific thrust

        # Create OSQP old version only works with old version of osqp
        # self.model = osqp.OSQP()
        # self.model.setup(P=self.P, A=self.A, l=np.zeros(nc), eps_rel=1e-4, eps_abs=1e-4, verbose=False)

        # Create OSQP
        self.model = osqp.OSQP()

        nx = self.N * (2 * ny + nu)
        nc = 2 * self.N * ny + self.N

        q0 = np.zeros(nx)
        l0 = np.zeros(nc)
        u0 = np.zeros(nc)  # or np.inf / something loose – they get overwritten later anyway

        self.model.setup(
            P=self.P, # quadratic matrix cost
            q=q0,# linear cost vector
            A=self.A, # constraint matrix
            l=l0, # lower bound on Ax
            u=u0,
            eps_rel=eps_rel_in,
            eps_abs=eps_abs_in,
            max_iter = 70,
            verbose=False,
        )

        # Manage linearization point
        self.T0 = 0 # mass-specific thrust
        self.Ibi = np.diag(1/Ib)

        # Logging for OSQP performance
        self.solve_times = []  # in milliseconds
        self.solve_iters = []  # iteration count per solve

    def codegen(self, dirname='uprightmpc2/gen'):
        try:
            self.model.codegen(dirname, project_type='', force_rewrite=True, parameters='matrices', FLOAT=True, LONG=False)
        except:
            # No worries if python module failed to compile
            pass

    def testDyn(self, T0sp, s0s, Btaus, y0, dy0):
        # Test
        self.A, l, u, Axidx = updateConstraint(self.N, self.A, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g, self.Tmax)
        xtest = openLoopX(self.N, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g)
        print((self.A @ xtest - l)[:2*self.N*ny])

    # def update1(self, T0sp, s0s, Btaus, y0, dy0, ydes, dydes):
    #     # Update
    #     self.A, self.l, self.u, self.Axidx = updateConstraint(self.N, self.A, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g, self.Tmax)
    #     self.Pdata, self.q = updateObjective(self.N, *self.Wts, ydes, dydes)
    #
    #     # OSQP solve ---
    #     self.model.update(Px=self.Pdata, Ax=self.A.data, q=self.q, l=self.l, u=self.u)
    #     # res = self.model.solve()
    #     # print(res)
    #     start = time.time()
    #     res = self.model.solve()
    #     end = time.time()
    #     elapsed = end - start
    #     print(f"OSQP solve time: {elapsed * 1e3:.3f} ms")
    #
    #     # print(f"OSQP iter per step: {res.info.iter}, status: {res.info.status}")
    #
    #     if 'solved' not in res.info.status:
    #         print(res.info.status)
    #     self.obj_val = res.info.obj_val
    #     # Functions for debugging
    #     self.obj = lambda x : 0.5 * x.T @ self.Pdense @ x + self.q.T @ x
    #     self.viol = lambda x : np.amin(np.hstack((self.A @ x - self.l, self.u - self.A @ x)))
    #     return res.x

    def update1(self, T0sp, s0s, Btaus, y0, dy0, ydes, dydes):
        # Update
        self.A, self.l, self.u, self.Axidx = updateConstraint(
            self.N, self.A, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g, self.Tmax
        )
        self.Pdata, self.q = updateObjective(self.N, *self.Wts, ydes, dydes)

        # OSQP solve ---
        self.model.update(Px=self.Pdata, Ax=self.A.data, q=self.q, l=self.l, u=self.u)

        # High-resolution timing for the solve
        t0 = time.perf_counter()
        res = self.model.solve()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1e3
        self.solve_times.append(elapsed_ms)
        self.solve_iters.append(res.info.iter)

        # Optional: print if you want live feedback
        # print(f"OSQP solve time: {elapsed_ms:.3f} ms, iters: {res.info.iter}")

        if 'solved' not in res.info.status:
            print(res.info.status)
        self.obj_val = res.info.obj_val
        # Functions for debugging
        self.obj = lambda x: 0.5 * x.T @ self.Pdense @ x + self.q.T @ x
        self.viol = lambda x: np.amin(np.hstack((self.A @ x - self.l, self.u - self.A @ x)))
        return res.x

    def update2(self, p0, R0, dq0, pdes, dpdes, sdes):
        # At current state
        s0 = np.copy(R0[:,2])
        s0s = [s0 for i in range(self.N)]
        Btau = (-R0 @ e3h @ self.Ibi)[:,:2] # no yaw torque
        Btaus = [Btau for i in range(self.N)]
        ds0 = -R0 @ e3h @ dq0[3:6] # omegaB

        y0 = np.hstack((p0, s0))
        dy0 = np.hstack((dq0[:3], ds0))
        ydes = np.hstack((pdes, sdes))
        dydes = np.hstack((dpdes, 0, 0, 0))

        self.prevsol = self.update1(self.T0, s0s, Btaus, y0, dy0, ydes, dydes)
        utilde = self.prevsol[2*ny*self.N : 2*ny*self.N+nu]
        self.T0 += utilde[0]

        return np.hstack((self.T0, utilde[1:]))

    def getAccDes(self, R0, dq0):
        dy1des = self.prevsol[ny*self.N : ny*self.N+ny] # from horiz
        # # Coordinate change for the velocity
        # bTw = lambda dq : np.hstack((R0.T @ dq[:3], dq[3:6]))
        dq1des = np.hstack((dy1des[:3], e3h @ R0.T @ dy1des[3:6])) # NOTE omegaz is lost
        # return (bTw(dq1des) - bTw(dq0)) / self.dt
        return (dq1des - dq0) / self.dt # return in world frame

    def update(self, p0, R0, dq0, pdes, dpdes, sdes, actualT0):
        if actualT0 >= 0:
            self.T0 = actualT0
        # Version of above that computes the desired body frame acceleration
        u = self.update2(p0, R0, dq0, pdes, dpdes, sdes)
        return u, self.getAccDes(R0, dq0)

    def solve(self):
        """
        Call pqp() with current Q, h, warm-start x, and settings.
        Returns an object with fields similar to OSQP (minimal subset).
        """
        if self.Q is None or self.h is None:
            raise RuntimeError("Call setup() before solve().")

        # Run the PQP algorithm; it returns new x and history of obj values
        x_opt, values = pqp(
            self.Q,
            self.h,
            x=self.x,           # warm start
            iters=self.iters,
            maxval=self.maxval
        )

        # Store warm start for the next call
        self.x = x_opt.copy()

        # Build a simple result object
        class Result:
            pass

        res = Result()
        res.x = x_opt
        res.obj_vals = values
        res.status = "solved"   # you could add real checks if desired

        return res
# assume ny, nu, e3h, initConstraint, updateConstraint, updateObjective, openLoopX
# are defined elsewhere as in your existing code

class UprightMPC2Other:
    def __init__(self, N, dt, g, TtoWmax,
                 ws, wds, wpr, wpf, wvr, wvf, wthrust, wmom, Ib,err_tol_in):

        self.N = N
        nx = self.N * (2*ny + nu)     # same as before
        nc = 2*self.N*ny + self.N

        # Build initial constraint sparsity & dummy P
        self.A, self.P = initConstraint(N, nx, nc)

        # Cost weights exactly as before
        Qyr  = np.hstack((np.full(3, wpr), np.full(3, ws)))
        Qyf  = np.hstack((np.full(3, wpf), np.full(3, ws)))
        Qdyr = np.hstack((np.full(3, wvr), np.full(3, wds)))
        Qdyf = np.hstack((np.full(3, wvf), np.full(3, wds)))
        R    = np.hstack((wthrust, np.full(2, wmom)))

        self.dt   = dt
        self.Wts  = [Qyr, Qyf, Qdyr, Qdyf, R]
        self.g    = g
        self.Tmax = TtoWmax * g
        self.T0   = 0.0
        self.Ibi  = np.diag(1.0 / Ib)

        # dimensions
        self.nx = nx
        self.nc = nc

        # logging
        self.solve_times = []
        self.solve_iters = []   # no real iteration count from qpsolvers, keep for API

        # warm start
        self.prevsol = None

        # store the error tolerance as a member
        self.err_tol = err_tol_in

    # ---- same testDyn as before, just calls updateConstraint/openLoopX ----
    def testDyn(self, T0sp, s0s, Btaus, y0, dy0):
        self.A, l, u, Axidx = updateConstraint(
            self.N, self.A, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g, self.Tmax
        )
        xtest = openLoopX(self.N, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g)
        print((self.A @ xtest - l)[:2*self.N*ny])

    # ---- core QP update+solve using qpsolvers + qpOASES backend ----
    def _solve_qp(self, H, g, A, lb, ub, lbA, ubA):
        """
        Wraps qpsolvers.solve_qp with qpOASES backend.

        H:   (nx, nx)
        g:   (nx,)
        A:   (nc, nx)  constraint matrix
        lb, ub:   (nx,)    box bounds on x
        lbA, ubA: (nc,)    OSQP-style range bounds  lbA <= A x <= ubA
        """
        t0 = time.perf_counter()

        # Convert range constraints to inequalities G x <= h
        # Take only finite bounds so we don't send infs to the solver.
        lbA = lbA.ravel()
        ubA = ubA.ravel()
        A_dense = np.asarray(A, dtype=float)

        upper_mask = np.isfinite(ubA)
        lower_mask = np.isfinite(lbA)

        G_list = []
        h_list = []

        if np.any(upper_mask):
            G_list.append(A_dense[upper_mask, :])
            h_list.append(ubA[upper_mask])

        if np.any(lower_mask):
            G_list.append(-A_dense[lower_mask, :])
            h_list.append(-lbA[lower_mask])

        if G_list:
            G = np.vstack(G_list)
            h = np.concatenate(h_list)
        else:
            G = None
            h = None

        # Warm start from previous solution if available
        initvals = self.prevsol if self.prevsol is not None else None

        # Solve QP: 0.5 x^T H x + g^T x
        # subject to G x <= h, lb <= x <= ub.
        xOpt = solve_qp(
            H, g,
            G=G, h=h,
            A=None, b=None,
            lb=lb, ub=ub,
            initvals=initvals,
            solver="qpoases",
            terminationTolerance = self.err_tol
            # this mimics options.setToMPC() in native qpOASES
            # predefined_options="MPC",
        )

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1e3
        self.solve_times.append(elapsed_ms)

        # qpsolvers doesn't expose nWSR; keep NaN placeholder
        self.solve_iters.append(np.nan)

        if xOpt is None:
            # fall back to previous solution or zeros if infeasible
            if self.prevsol is not None:
                xOpt = self.prevsol.copy()
            else:
                xOpt = np.zeros(self.nx)

        return xOpt

    def update1(self, T0sp, s0s, Btaus, y0, dy0, ydes, dydes):
        # 1) update A, l, u exactly like in OSQP version
        self.A, self.l, self.u, self.Axidx = updateConstraint(
            self.N, self.A, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g, self.Tmax
        )

        # 2) update cost (Pdata = diag entries, q = linear term)
        self.Pdata, self.q = updateObjective(self.N, *self.Wts, ydes, dydes)

        # 3) build dense matrices for qpOASES via qpsolvers
        H = np.diag(self.Pdata.astype(float))              # (nx, nx)
        g = self.q.astype(float).copy()                    # (nx,)

        # No explicit bounds on x -> big box
        BIG = 1e20
        lb  = -BIG * np.ones(self.nx)
        ub  =  BIG * np.ones(self.nx)

        # Constraint bounds are exactly l, u
        lbA = self.l.astype(float)
        ubA = self.u.astype(float)

        # 4) solve with qpsolvers + qpOASES backend
        xOpt = self._solve_qp(H, g, self.A.toarray(), lb, ub, lbA, ubA)

        self.obj_val = 0.5 * xOpt @ (H @ xOpt) + g @ xOpt
        self.prevsol = xOpt
        return xOpt

    # ---- same as your current update2 / getAccDes / update ----
    def update2(self, p0, R0, dq0, pdes, dpdes, sdes):
        s0   = np.copy(R0[:, 2])
        s0s  = [s0 for _ in range(self.N)]
        Btau = (-R0 @ e3h @ self.Ibi)[:, :2]  # no yaw torque
        Btaus = [Btau for _ in range(self.N)]
        ds0  = -R0 @ e3h @ dq0[3:6]

        y0   = np.hstack((p0, s0))
        dy0  = np.hstack((dq0[:3], ds0))
        ydes  = np.hstack((pdes, sdes))
        dydes = np.hstack((dpdes, 0, 0, 0))

        xOpt = self.update1(self.T0, s0s, Btaus, y0, dy0, ydes, dydes)

        utilde = xOpt[2*ny*self.N : 2*ny*self.N+nu]
        self.T0 += utilde[0]
        return np.hstack((self.T0, utilde[1:]))

    def getAccDes(self, R0, dq0):
        dy1des = self.prevsol[ny*self.N : ny*self.N+ny]
        dq1des = np.hstack((dy1des[:3], e3h @ R0.T @ dy1des[3:6]))
        return (dq1des - dq0) / self.dt

    def update(self, p0, R0, dq0, pdes, dpdes, sdes, actualT0):
        if actualT0 >= 0:
            self.T0 = actualT0
        u = self.update2(p0, R0, dq0, pdes, dpdes, sdes)
        return u, self.getAccDes(R0, dq0)

class UprightMPC2qpOases:
    def __init__(self, N, dt, g, TtoWmax,
                 ws, wds, wpr, wpf, wvr, wvf, wthrust, wmom, Ib, err_tol_in):

        self.N = N
        nx = self.N * (2 * ny + nu)
        nc = 2 * self.N * ny + self.N

        # Constraint structure and dummy P (sparsity only)
        self.A, self.P = initConstraint(N, nx, nc)

        # Cost weights (same pattern as UprightMPC2 / UprightMPC2_Other)
        Qyr  = np.hstack((np.full(3, wpr), np.full(3, ws)))
        Qyf  = np.hstack((np.full(3, wpf), np.full(3, ws)))
        Qdyr = np.hstack((np.full(3, wvr), np.full(3, wds)))
        Qdyf = np.hstack((np.full(3, wvf), np.full(3, wds)))
        R    = np.hstack((wthrust, np.full(2, wmom)))

        self.dt   = dt
        self.Wts  = [Qyr, Qyf, Qdyr, Qdyf, R]
        self.g    = g
        self.Tmax = TtoWmax * g
        self.T0   = 0.0
        self.Ibi  = np.diag(1.0 / Ib)

        self.nx = nx
        self.nc = nc

        # logging
        self.solve_times = []
        self.solve_iters = []   # qpOASES nWSR not exposed here; keep NaN placeholder

        # warm start
        self.prevsol = None

        # store error tolerance (used as terminationTolerance in qpOASES)
        self.err_tol = err_tol_in

    def testDyn(self, T0sp, s0s, Btaus, y0, dy0):
        # Same test as in other controllers
        self.A, l, u, Axidx = updateConstraint(
            self.N, self.A, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g, self.Tmax
        )
        xtest = openLoopX(self.N, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g)
        print((self.A @ xtest - l)[:2 * self.N * ny])

    def _solve_qp(self, H, g, A, lbA, ubA):
        """
        Wrapper around casadi_qpsolve (CasADi + qpOASES):

            min 0.5 x^T H x + g^T x
            s.t. lbA <= A x <= ubA
                 lbx <= x <= ubx    (here we use wide box bounds)
        """
        t0 = time.perf_counter()

        # Box bounds: effectively unbounded
        BIG = 1e20
        lbx = -BIG * np.ones(self.nx)
        ubx =  BIG * np.ones(self.nx)

        # Ensure dense ndarray for CasADi
        A_dense = np.asarray(A, dtype=float)
        lbA = np.asarray(lbA, dtype=float).ravel()
        ubA = np.asarray(ubA, dtype=float).ravel()

        try:
            x_opt = casadi_qpsolve(
                H,
                g,
                lbx,
                ubx,
                A=A_dense,
                lba=lbA,
                uba=ubA,
                termination_tol=self.err_tol,
                verbose=False,
            )
        except Exception as e:
            print("[UprightMPC2qpOases] qpOASES solve failed:", repr(e))

            # Fallback if qpOASES fails (keep controller alive)
            if self.prevsol is not None:
                x_opt = self.prevsol.copy()
            else:
                x_opt = np.zeros(self.nx)

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1e3
        self.solve_times.append(elapsed_ms)
        self.solve_iters.append(np.nan)  # qpOASES iteration count not available here

        return x_opt

    def update1(self, T0sp, s0s, Btaus, y0, dy0, ydes, dydes):
        # 1) update constraint matrix and bounds
        self.A, self.l, self.u, self.Axidx = updateConstraint(
            self.N, self.A, self.dt, T0sp, s0s, Btaus, y0, dy0, self.g, self.Tmax
        )

        # 2) update objective (returns diagonal entries and linear term)
        self.Pdata, self.q = updateObjective(self.N, *self.Wts, ydes, dydes)

        # 3) build dense Hessian and gradient for qpOASES
        H = np.diag(self.Pdata.astype(float))   # (nx, nx)
        g = self.q.astype(float).copy()         # (nx,)

        # 4) A x is constrained between l and u
        lbA = self.l
        ubA = self.u

        # 5) solve QP using CasADi + qpOASES
        x_opt = self._solve_qp(H, g, self.A.toarray(), lbA, ubA)

        # store solution, objective value, etc.
        self.prevsol = x_opt
        self.obj_val = 0.5 * x_opt @ (H @ x_opt) + g @ x_opt

        return x_opt

    def update2(self, p0, R0, dq0, pdes, dpdes, sdes):
        # Same structure as other MPC classes
        s0   = np.copy(R0[:, 2])
        s0s  = [s0 for _ in range(self.N)]
        Btau = (-R0 @ e3h @ self.Ibi)[:, :2]  # no yaw torque
        Btaus = [Btau for _ in range(self.N)]
        ds0  = -R0 @ e3h @ dq0[3:6]

        y0   = np.hstack((p0, s0))
        dy0  = np.hstack((dq0[:3], ds0))
        ydes  = np.hstack((pdes, sdes))
        dydes = np.hstack((dpdes, 0, 0, 0))

        x_opt = self.update1(self.T0, s0s, Btaus, y0, dy0, ydes, dydes)

        utilde = x_opt[2 * ny * self.N : 2 * ny * self.N + nu]
        self.T0 += utilde[0]
        return np.hstack((self.T0, utilde[1:]))

    def getAccDes(self, R0, dq0):
        dy1des = self.prevsol[ny * self.N : ny * self.N + ny]
        dq1des = np.hstack((dy1des[:3], e3h @ R0.T @ dy1des[3:6]))
        return (dq1des - dq0) / self.dt

    def update(self, p0, R0, dq0, pdes, dpdes, sdes, actualT0):
        if actualT0 >= 0:
            self.T0 = actualT0
        u = self.update2(p0, R0, dq0, pdes, dpdes, sdes)
        return u, self.getAccDes(R0, dq0)

def createMPC(N=3, ws=1e1, wds=1e3, wpr=1, wvr=1e3, wpf=5, wvf=2e3,
              wthrust=1e-1, wmom=1e-2, TtoWmax=2, popts=np.zeros(90),
              solver = "OSQP",eps_rel=1e-4,eps_abs=1e-4, **kwargs):
    """Returns the mdl. Parameters are
    N: prediction horizon length
    wpr: running position weight. First three components of Qyr
    wpf: final position weight. First three components of Qyf.
    ws : running orientation-vector weight. Last three components of Qyr.
    wds: running orientation-rate weight. Last three components of Qdyr (Q_dot_yr).
    wvr: running velocity weight. First three components of Qdyr.
    wvf: final velocity weight. First three components of Qdyf.

    Remember "we frequently use a higher final cost than running cost," see paper.
    """


    dt = 5
    g = 9.81e-3
    # WLQP inputs
    mb = 100
    # what "u" is depends on w(u). Here in python testing with w(u) = [0,0,u0,u1,u2,u3].
    # Setting first 2 elements of Qw -> 0 => should not affect objective as longs as dumax does not constrain.
    Qw = np.hstack((np.zeros(2), np.zeros(4)))
    umin = np.array([0, -0.5, -0.2, -0.1])
    umax = np.array([10, 0.5, 0.2, 0.1])
    dumax = np.array([10, 10, 10, 10]) # /s
    # # WLQP stuff - copied from isolated C implementation
    # umin = np.array([50, -0.5, -0.2, -0.1])
    # umax = np.array([240, -0.5, -0.2, -0.1])
    # dumax = np.array([5e3, 10, 10, 10]) # /s
    controlRate = 1000
    if solver == "OSQP":
        pyver = UprightMPC2(N, dt, g, TtoWmax, ws, wds, wpr, wpf, wvr, wvf, wthrust, wmom, Ib.diagonal(),eps_rel,eps_abs)
    elif solver == "qpOASES":
        # Use CasADi + qpOASES-based MPC
        pyver = UprightMPC2qpOases(
            N, dt, g, TtoWmax,
            ws, wds, wpr, wpf, wvr, wvf,
            wthrust, wmom, Ib.diagonal(),
            eps_abs  # use eps_abs as termination tol
        )
    else:
        pyver = UprightMPC2Other(N, dt, g, TtoWmax, ws, wds, wpr, wpf, wvr, wvf, wthrust, wmom, Ib.diagonal(),eps_abs)
    # C version can be tested too
    cver = UprightMPC2C(dt, g, TtoWmax, ws, wds, wpr, wpf, wvr, wvf, wthrust, wmom, Ib.diagonal(), 50)
    return pyver, cver

def reactiveController(p, Rb, dq, pdes, kpos=[5e-3,5e-1], kz=[1e-1,1e0], ks=[10e0,1e2], **kwargs):
    # Pakpong-style reactive controller
    sdes = np.clip(kpos[0] * (pdes - p) - kpos[1] * dq[:3], np.full(3, -0.5), np.full(3, 0.5))
    sdes[2] = 1
    # sdes = np.array([0,0,1])
    omega = dq[3:]
    dp = dq[:3]
    s = Rb[:,2]
    ds = -Rb @ e3h @ omega
    # Template controller <- LATEST
    fz = kz[0] * (pdes[2] - p[2]) - kz[1] * dq[2]
    fTorn = ks[0] * (s - sdes) + ks[1] * ds
    fTorn[2] = 0
    fAorn = -e3h @ Rb.T @ fTorn
    return np.hstack((fz, fAorn[:2]))

if __name__ == "__main__":
    up, upc = createMPC()
    # Dyn test
    T0 = 0.5
    s0s = [[0.1,0.1,0.9] for i in range(up.N)]
    Btaus = [np.full((3,2),1.123) for i in range(up.N)]
    y0 = np.random.rand(6)
    dy0 = np.random.rand(6)
    ydes = np.zeros_like(y0)
    dydes = np.zeros_like(y0)
    up.testDyn(T0, s0s, Btaus, y0, dy0)

    # FIXME: test
    p = np.random.rand(3)
    R = np.random.rand(3, 3)
    dq = np.random.rand(6)
    pdes = np.random.rand(3)
    dpdes = np.random.rand(3)
    sdes = np.random.rand(3)
    retc = upc.update(p, R, dq, pdes, dpdes, sdes)
    cl, cu, cq = upc.vectors()
    cP, cAdata, cAidx = upc.matrices()
    ret = up.update(p, R, dq, pdes, dpdes, sdes)
    # print(cAdata - up.A.data[cAidx])#OK
    # print(cAidx - up.Axidx)#OK
    # print(cl - up.l)#OK
    # print(cu - up.u)#OK
    # print(cq - up.q)#OK
    # print(cP - up.Pdata)#OK
    print(ret[0], ret[1], ret[0]-retc[0], ret[1]-retc[1])
