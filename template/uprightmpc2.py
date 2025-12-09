from pickle import FALSE

import numpy as np
from scipy.spatial.transform import Rotation
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from genqp import quadrotorNLDyn
from template_controllers import createMPC, reactiveController
import flight_tasks
from time import perf_counter
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# mpl.use('macosx')
# mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

import progressbar
from plot_helpers import *
import time

# def viewControlTestLog(log, log2=None, callShow=True, goal0=False, desTraj=False, vscale=0.4):
#     def posParamPlot(_ax):
#         traj3plot(_ax, log['t'], log['y'][:,:3], log['y'][:,3:6], "Blues_r", vscale=vscale)
#         aspectEqual3(_ax, log['y'][:,:3])
#         if log2 is not None:
#             traj3plot(_ax, log2['t'], log2['y'][:,:3], log2['y'][:,3:6], "Reds_r", vscale=vscale)
#         # _ax.plot(log['t'], log['pdes'][:,0], 'k--', alpha=0.3)
#         if goal0:
#             _ax.plot([0], [0], [0], 'g*', markersize=10, zorder=10)
#             _ax.legend(('MPC', 'Reactive', 'Goal'))
#         else:
#             _ax.legend(('MPC', 'Reactive'))
#         if desTraj:
#             _ax.plot(log['pdes'][:,0], log['pdes'][:,1], log['pdes'][:,2], 'k--', alpha=0.5, zorder=9)
#         _ax.set_xlabel('x [mm]')
#         _ax.set_ylabel('y [mm]')
#         _ax.set_zlabel('z [mm]')

def viewControlTestLog(
    log,
    log2=None,
    callShow=True,
    goal0=False,
    desTraj=False,
    vscale=0.4,
    xlabel='x [mm]',
    ylabel='y [mm]',
    zlabel='z [mm]',
    label_fontsize=12,
    tick_fontsize=10,
    label_pad = 10,
):

    def posParamPlot(_ax):
        # Main trajectories
        traj3plot(_ax, log['t'], log['y'][:, :3], log['y'][:, 3:6], "Blues_r", vscale=vscale)
        aspectEqual3(_ax, log['y'][:, :3])

        if log2 is not None:
            traj3plot(_ax, log2['t'], log2['y'][:, :3], log2['y'][:, 3:6], "Reds_r", vscale=vscale)

        # Goal marker
        if goal0:
            _ax.plot([0], [0], [0], 'g*', markersize=10, zorder=10)

        # Desired trajectory
        if desTraj:
            _ax.plot(
                log['pdes'][:, 0],
                log['pdes'][:, 1],
                log['pdes'][:, 2],
                'k--',
                alpha=0.5,
                zorder=9,
                label='Desired trajectory'
            )

        # Build legend labels
        legend_labels = ['MPC']
        if log2 is not None:
            legend_labels.append('Reactive')
        if goal0:
            legend_labels.append('Goal')
        if desTraj:
            legend_labels.append('Desired trajectory')

        _ax.legend(legend_labels)

        # Axis labels with configurable fontsize
        _ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=label_pad)
        _ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=label_pad)
        _ax.set_zlabel(zlabel, fontsize=label_fontsize, labelpad=label_pad)

        # Tick label fontsize
        _ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # ... rest of your function that creates the figure/axes and calls posParamPlot(...)

    def posPlot(_ax):
        _ax.plot(log['t'], log['y'][:,:3])
        if log2 is not None:
            _ax.plot(log2['t'], log2['y'][:,:3], '--')
        _ax.plot(log['t'], log['pdes'][:,0], 'k--', alpha=0.3)
        _ax.set_ylabel('p')
    def splot(_ax):
        _ax.plot(log['t'], log['y'][:,3:6])
        _ax.axhline(y=0, color='k', alpha=0.3)
        _ax.set_ylabel('s')
    def inputsPlot(_ax1, _ax2):
        _ax1.plot(log['t'], log['u'][:,0])
        _ax1.axhline(y=0, color='k', alpha=0.3)
        _ax1.set_ylabel('Sp. thrust')
        _ax2.plot(log['t'], log['u'][:,1:])
        _ax2.axhline(y=0, color='k', alpha=0.3)
        _ax2.set_ylabel('Moments')
    def velsPlot(_ax1, _ax2):
        _ax1.plot(log['t'], log['y'][:,6:9])
        _ax1.axhline(y=0, color='k', alpha=0.3)
        _ax1.set_ylabel('v')
        if _ax2 is not None:
            _ax2.plot(log['t'], log['y'][:,9:12])
            _ax2.axhline(y=0, color='k', alpha=0.3)
            _ax2.set_ylabel('omega')
    def accdesPlots(_ax1, _ax2):
        _ax1.plot(log['t'], log['accdes'][:,:3])
        _ax1.axhline(y=0, color='k', alpha=0.3)
        _ax1.set_ylabel('accdes pos')
        _ax2.plot(log['t'], log['accdes'][:,3:])
        _ax2.axhline(y=0, color='k', alpha=0.3)
        _ax2.set_ylabel('accdes ang')
    def wlqpuPlots(_ax):
        _ax.plot(log['t'], log['wlqpu'][:,:2])
        _ax.plot(log['t'], log['wlqpu'][:,2:],'--')
        _ax.legend(('0','1','2','3'))
        _ax.set_ylabel('wlqpu')
    
    # fig = plt.figure()
    # ax = [fig.add_subplot(3,3,i+1) for i in range(1,12)]
    # posPlot(ax[0])
    # splot(ax[1])
    # inputsPlot(ax[2], ax[3])
    # velsPlot(ax[4], ax[5])
    # accdesPlots(ax[6], ax[7])
    # wlqpuPlots(ax[8])
    # fig.tight_layout()
    
    fig = plt.figure(dpi=250)
    ax3d = fig.add_subplot(1,1,1,projection='3d')
    posParamPlot(ax3d)

    if callShow:
        plt.show()

def controlTest(mdl, tend, dtsim=0.2, hlInterval=None, useMPC=True, trajFreq=0, trajAmp=0, ascentIC=False, showPlots=True, tpert=None, speedTest=False, perchTraj=False, flipTask=False, taulim=100, **kwargs):
    """
    You will hopefully have set a mdl with createMPC
    (which itself calls UprightMPC2...). If not, this will
    make an MPC for you.

    trajFreq in Hz, trajAmp in mm"""
    if mdl is None and useMPC:
        mdl, _ = createMPC(**kwargs)
    speedTestvdes = 2 # m/s
    speedTestdur = 500
    # Initial conditions
    dq = np.zeros(6)
    if ascentIC or speedTest or perchTraj:
        p = np.array([0, 0, -50]) if ascentIC else np.array([-speedTestdur*speedTestvdes, 0, 0])
        if perchTraj:
            p = np.array([-100, 0, 0])
        Rb = np.eye(3)
    else:
        p = np.array([0, 0, 0])
        Rb = np.eye(3) if flipTask else Rotation.from_euler('xyz', [0.5,-0.5,0]).as_matrix()
        dq[0] = 0.1
    pdes = np.zeros(3)
    dpdes = np.zeros(3)
    sdes = np.array([0,0,1])
    initialPos = np.copy(p)
    
    tt = np.arange(tend, step=dtsim)
    Nt = len(tt)

    # for logging
    log = {'t': tt, 'y': np.zeros((Nt, 12)), 'u': np.zeros((Nt, 3)), 'pdes': np.zeros((Nt, 3)), 'accdes': np.zeros((Nt,6))}

    ddqdes = None # test integrate ddq sim below
    avgTime = 0.0
    uquad = np.zeros(3)
    thlPrev = 0

    for ti in range(Nt):
        # Traj to follow
        if flipTask:
            pdes, dpdes, sdes = flight_tasks.flip(tt[ti], initialPos)
        elif perchTraj:
            pdes, dpdes, sdes = flight_tasks.perch(tt[ti], initialPos)
        elif speedTest:
            pdes, dpdes, sdes = flight_tasks.straightAcc(tt[ti], initialPos, vdes=speedTestvdes, tduration=speedTestdur)
        else:
            pdes, dpdes, sdes = flight_tasks.helix(tt[ti], initialPos, trajAmp=trajAmp, trajFreq=trajFreq, dz=0.1, useY=True) # if useY is false, then it just moves back and forth while going up
            # Add perturbation for this traj
            if tpert is not None and tt[ti] > tpert:
                dq[1] += 2
                tpert = None

        # Call HL controller
        if hlInterval is None or tt[ti] - thlPrev > hlInterval:
            if useMPC:
                t1 = perf_counter()
                actualT0 = -1.0  # seems like there's an error with not enough arguments below??
                uquad, log['accdes'][ti,:] = mdl.update(p, Rb, dq, pdes, dpdes, sdes,actualT0)
                avgTime += 0.01 * (perf_counter() - t1 - avgTime)
                # # Alternate simulation by integrating accDes
                # ddqdes = accdess[ti,:]
            else:
                uquad = reactiveController(p, Rb, dq, pdes, **kwargs)
            thlPrev = tt[ti]
        # u = np.array([1,0.1,0])
        # Input limit
        for i in range(2):
            uquad[i+1] = np.clip(uquad[i+1], -taulim, taulim)

        p, Rb, dq = quadrotorNLDyn(p, Rb, dq, uquad, dtsim, ddq=ddqdes)
        log['y'][ti,:] = np.hstack((p, Rb[:,2], dq))
        log['u'][ti,:] = uquad
        log['pdes'][ti,:] = pdes
    if useMPC and showPlots:
        print("Time (ms):", avgTime * 1e3)
    if showPlots:
        viewControlTestLog(log)
    return log

def logMetric(log):
    # A metric to plot about how good the tracking was
    Nt = len(log['t'])
    perr = log['y'][:, :3] - log['pdes']
    tau = log['u'][:, 1:3]
    serr = log['y'][:, 3:6]
    serr[:, 2] -= 1.0
    err = 0
    eff = 0
    for i in range(Nt):
        err += np.dot(perr[i, :], perr[i, :])  # + 10 * np.dot(serr[i,:], serr[i,:])
        eff += np.dot(tau[i, :], tau[i, :])  # + 10 * np.dot(serr[i,:], serr[i,:])
    err = np.sqrt(err / (3 * Nt)) # this gives RMS error.
    eff /= Nt
    return err, eff


# def logMetric(log):
#     # A metric to plot about how good the tracking was
#     Nt = len(log['t'])
#     perr = log['y'][:,:3]
#     tau = log['u'][:,1:3]
#     serr = log['y'][:,3:6]
#     serr[:,2] -= 1.0
#     err = 0
#     eff = 0
#     for i in range(Nt):
#         err += np.dot(perr[i,:], perr[i,:])# + 10 * np.dot(serr[i,:], serr[i,:])
#         eff += np.dot(tau[i,:], tau[i,:])# + 10 * np.dot(serr[i,:], serr[i,:])
#     err /= Nt
#     eff /= Nt
#     return err, eff

def papPlots(bmpc):
    """Baseline mpc as argument"""
    def flipTask():
        l1 = controlTest(bmpc, 1000, useMPC=True, showPlots=False, flipTask=True)
        viewControlTestLog(l1, desTraj=True, vscale=10)
        fig, ax = plt.subplots(1,3, figsize=(7.5,2.5))
        for i in range(0,3,2):
            ax[i].plot(1e-3*l1[ 't'], l1['y'][:,i], 'b')
            ax[i].plot(1e-3*l1['t'], l1['pdes'][:,i], 'k--', alpha=0.3)
            ax[i].set_xlabel('t [s]')
            
        ax[1].plot(1e-3*l1['t'], 180/np.pi*np.arctan2(l1['y'][:,3], l1['y'][:,5]), 'b')
        ax[1].plot([0, 0.1, 0.2], [0, 0, -180], 'k--', alpha=0.3)
        ax[1].plot([0.2, 0.3, 1], [180, 0, 0], 'k--', alpha=0.3)
        ax[1].set_ylabel('Angle [deg]')
        ax[0].set_ylabel('x [mm]')
        ax[2].set_ylabel('z [mm]')
        fig.tight_layout()
        plt.show()

    def perchTask():
        l1 = controlTest(bmpc, 550, useMPC=True, showPlots=False, perchTraj=True)
        viewControlTestLog(l1, desTraj=True, vscale=10)
        fig, ax = plt.subplots(1,3, figsize=(7.5,2.5))
        for i in range(0,3,2):
            ax[i].plot(1e-3*l1['t'], l1['y'][:,i], 'b')
            ax[i].plot(1e-3*l1['t'], l1['pdes'][:,i], 'k--', alpha=0.3)
            ax[i].set_xlabel('t [s]')
            
        ax[1].plot(1e-3*l1['t'], 180/np.pi*np.arctan2(l1['y'][:,3], l1['y'][:,5]), 'b')
        ax[1].plot([0, 0.45, 0.55], [0, 0, -90], 'k--', alpha=0.3)
        ax[1].set_ylabel('Angle [deg]')
        ax[0].set_ylabel('x [mm]')
        ax[2].set_ylabel('z [mm]')
        fig.tight_layout()
        plt.show()

    def hoverTask(show3d, reactiveArgs1, reactiveArgs2=None):
        l1 = controlTest(bmpc, 300, useMPC=True, showPlots=False)
        l2 = controlTest(bmpc, 1000, useMPC=False, showPlots=False, **reactiveArgs1)
        if reactiveArgs2 is not None:
            l3 = controlTest(bmpc, 1000, useMPC=False, showPlots=False, **reactiveArgs2)
        if show3d:
            viewControlTestLog(l1, log2=l2, goal0=True)
        else:
            fig, ax = plt.subplots(1,2, figsize=(5,2.5))
            for i in range(2):
                ax[i].plot(1e-3*l1['t'], l1['y'][:,i], 'b')
                ax[i].plot(1e-3*l2['t'], l2['y'][:,i], 'r')
                if reactiveArgs2 is not None:
                    ax[i].plot(1e-3*l3['t'], l3['y'][:,i], 'r--')
                ax[i].plot(1e-3*l2['t'], l2['pdes'][:,i], 'k--', alpha=0.3)
                ax[i].set_xlabel('t [s]')
            ax[0].set_ylabel('x [mm]')
            ax[1].set_ylabel('y [mm]')
            fig.tight_layout()
            plt.show()

    def sTask(reactiveArgs1, reactiveArgs2=None):
        l1 = controlTest(bmpc, 2000, useMPC=True, showPlots=False, trajAmp=50, trajFreq=1, tpert=1000)
        l2 = controlTest(bmpc, 2000, useMPC=False, showPlots=False, trajAmp=50, trajFreq=1, tpert=1000, **reactiveArgs1)
        if reactiveArgs2 is not None:
            l3 = controlTest(bmpc, 2000, useMPC=False, showPlots=False, trajAmp=50, trajFreq=1, tpert=1000, **reactiveArgs2)
        viewControlTestLog(l1, log2=l2, desTraj=True, vscale=20)
        fig, ax = plt.subplots(1,2, figsize=(5,2.5))
        for i in range(2):
            ax[i].plot(1e-3*l1['t'], 1e-3*l1['y'][:,i], 'b')
            ax[i].plot(1e-3*l2['t'], 1e-3*l2['y'][:,i], 'r')
            if reactiveArgs2 is not None:
                ax[i].plot(1e-3*l3['t'], l3['y'][:,i], 'r--')
            ax[i].plot(1e-3*l2['t'], 1e-3*l2['pdes'][:,i], 'k--', alpha=0.3)
            ax[i].set_xlabel('t [s]')
        ax[0].set_ylabel('x [m]')
        ax[1].set_ylabel('y [m]')
        fig.tight_layout()
        plt.show()

    def accTask(reactiveArgs):
        l1 = controlTest(bmpc, 1000, useMPC=True, showPlots=False, speedTest=True)
        l2 = controlTest(bmpc, 1000, useMPC=False, showPlots=False, speedTest=True, **reactiveArgs)
        viewControlTestLog(l1, log2=l2, desTraj=True, vscale=100, goal0=True)
        fig, ax = plt.subplots(1,2, figsize=(5,2.5))
        ax[0].plot(1e-3*l1['t'], 1e-3*l1['y'][:,0], 'b')
        ax[0].plot(1e-3*l2['t'], 1e-3*l2['y'][:,0], 'r')
        ax[0].plot(1e-3*l2['t'], 1e-3*l2['pdes'][:,0], 'k--', alpha=0.3)
        ax[0].set_ylabel('x [m]')
        ax[0].set_xlabel('t [s]')
        ax[1].plot(1e-3*l1['t'], l1['y'][:,6], 'b')
        ax[1].plot(1e-3*l2['t'], l2['y'][:,6], 'r')
        ax[1].set_ylabel('xdot [m/s]')
        ax[1].set_xlabel('t [s]')
        fig.tight_layout()
        plt.show()

    # Hover tuning ---------
    def gainTuningSims(useMPC, kwgain, k1range, k2range, kwfixedn, kwfixedv, npts=10):
        k1s = np.linspace(*k1range,num=npts)
        k2s = np.linspace(*k2range,num=npts)
        xv, yv = np.meshgrid(k1s, k2s, indexing='ij') # treat xv[i,j], yv[i,j]
        costs = np.zeros_like(xv)
        efforts = np.zeros_like(xv)
        # create a progress bar
        widgets = [
            'Progress: ', progressbar.Percentage(),
            ' ', progressbar.Bar(),
            ' ', progressbar.ETA(),
        ]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=np.prod(costs.shape))
        nrun = 0
        for i in range(len(k1s)):
            for j in range(len(k2s)):
                nrun += 1
                bar.update(nrun)
                try:
                    if useMPC:
                        kwgain = 'mpc_wpos' # overwrite filename
                        # Assume same ratio of final to running cost
                        kwargs = {'wpr': xv[i,j], 'wvr': yv[i,j], 'wpf': 5*xv[i,j], 'wvf': 2*yv[i,j]}
                    else: # Reactive
                        kwargs = {kwgain: [xv[i,j],yv[i,j]], kwfixedn: kwfixedv}
                    l2 = controlTest(None, 1000, useMPC=useMPC, showPlots=False, taulim=10, **kwargs)
                    costs[i,j], efforts[i,j] = logMetric(l2)
                except KeyboardInterrupt:
                    raise
                except:
                    costs[i,j] = efforts[i,j] = np.nan
        np.savez(kwgain+str('.npz'), xv=xv, yv=yv, costs=costs, efforts=efforts)

    def gainTuningPlots(maxcost=10):
        lmpc = controlTest(bmpc, 1000, useMPC=True, showPlots=False)
        empc, effmpc = logMetric(lmpc)
                        
        def plot1(ax, dat, cbar=False):
            costs = np.clip(dat['costs'] / empc, 0, maxcost)
            im = ax.pcolormesh(dat['xv'], dat['yv'], costs, cmap='gray_r', shading='auto', vmin=0, vmax=maxcost)
            if cbar:
                fig.colorbar(im)
        
        fig, ax = plt.subplots(1,3,figsize=(12,4))
        plot1(ax[0], np.load('ks.npz'))
        ax[0].plot([15], [100], 'r*', ms=20)
        plot1(ax[1], np.load('kpos.npz'))
        ax[1].plot([0.01, 0.04], [1.0, 1.25], 'r*', ms=20)
        plot1(ax[2], np.load('mpc_wpos.npz'), cbar=True)
        ax[2].plot([1], [1e3], 'b*', ms=20)
        plt.show()

    def trackingEffortPlot(files):
        # Baseline
        lmpc = controlTest(bmpc, 1000, useMPC=True, showPlots=False)
        empc, effmpc = logMetric(lmpc)
        costs2 = []
        effs2 = []
        costs2mpc = []
        effs2mpc = []
        for fname in files:
            dat = np.load(fname)
            costs = dat['costs'].ravel() / empc
            effs = dat['efforts'].ravel() / effmpc
            ii = np.where(costs < 10)[0]
            if 'mpc' in fname:
                costs2mpc.append(costs[ii])
                effs2mpc.append(effs[ii])
            else:
                costs2.append(costs[ii])
                effs2.append(effs[ii])
        fig, ax = plt.subplots(1, figsize=(4,4))
        if len(costs2mpc) > 0:
            ax.scatter(costs2mpc, effs2mpc, color='b', label='MPC')
        ax.scatter(costs2, effs2, color='r', label='Reactive')
        ax.axhline(1, color='k', linestyle='dashed', alpha=0.3)
        ax.axvline(1, color='k', linestyle='dashed', alpha=0.3)
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')
        ax.set_xlabel('Relative tracking error [ ]')
        ax.set_ylabel('Relative actuator effort [ ]')
        ax.legend()
        plt.show()

    # Tuning ---------------
    # # Run and save data
    # # defaults kpos=[5e-3,5e-1], kz=[1e-1,1e0], ks=[10e0,1e2]
    # gainTuningSims(False, 'ks', [5e0,2e1], [2e1,2e2], 'kpos', [5e-3,5e-1])
    # gainTuningSims(False, 'kpos', [1e-3,8e-2], [1e-1,2e0], 'ks', [15,100])
    # # defaults wpr=1, wvr=1e3, wpf=5, wvf=2e3
    # gainTuningSims(True, 'wpos', [0.5,10], [0.5e3, 10e3], None, None)
    
    # hoverTask(False, {'ks':[15,100], 'kpos':[0.01,1]}, {'ks':[15,100], 'kpos':[0.04,1.25]})
    # sTask({'ks':[15,100], 'kpos':[0.01,1]})

    # gainTuningPlots()
    # trackingEffortPlot(['mpc_wpos.npz','kpos.npz'])

    # sim1hover -------------------
    # hoverTask(False, {'ks':[15,100], 'kpos':[0.01,1]})
    # sTask({'ks':[15,100], 'kpos':[0.01,1]})
    # accTask({'ks':[15,100], 'kpos':[0.01,1]})
    # sim1perch --------------
    # flipTask()
    # perchTask()

def run_sweep_and_save():
    vals = np.logspace(-5, 0, 50)   # 50 log-spaced tolerances 1e-5 ... 1
    avg_times_ms = []
    rms_errors   = []

    for err_tol in vals:
        print(f"\n=== Trial eps_rel=eps_abs={err_tol:.2e} ===")

        up, upc = createMPC(
            solver = "OSQP",
            eps_rel=err_tol,
            eps_abs=err_tol,
        )

        start = time.time()
        log = controlTest(
            up,
            tend=4000,
            useMPC=True,
            trajAmp=100,
            trajFreq=1,
            hlInterval=5,
            tpert=2000,
            showPlots=False,
        )
        end = time.time()

        if hasattr(up, "solve_times") and len(up.solve_times) > 0:
            avg_time_ms = 1e3 * np.mean(up.solve_times)
        else:
            num_solves = max(1, int(np.ceil(4000 / 5.0)))
            avg_time_ms = 1e3 * (end - start) / num_solves

        RMS_err, _ = logMetric(log)

        avg_times_ms.append(avg_time_ms)
        rms_errors.append(RMS_err)

        print(f"  avg time: {avg_time_ms:.4f} ms,  RMS err: {RMS_err:.4f}")

    avg_times_ms = np.array(avg_times_ms)
    rms_errors   = np.array(rms_errors)

    # ★ Save everything to disk
    np.savez(
        DATA_FILE,
        vals=vals,
        avg_times_ms=avg_times_ms,
        rms_errors=rms_errors,
    )
    print(f"\nSaved results to {DATA_FILE}")

def run_sweep_and_save_non_osqp(DATA_FILE):
    vals = np.logspace(-5, 0, 50)   # 50 log-spaced tolerances 1e-5 ... 1
    avg_times_ms = []
    rms_errors   = []

    for err_tol in vals:
        print(f"\n=== Trial eps_rel=eps_abs={err_tol:.2e} ===")

        up, upc = createMPC(
            solver = "qpOASES",
            eps_abs = err_tol,
        )

        start = time.time()
        log = controlTest(
            up,
            tend=4000,
            useMPC=True,
            trajAmp=100,
            trajFreq=1,
            hlInterval=5,
            tpert=2000,
            showPlots=False,
        )
        end = time.time()

        if hasattr(up, "solve_times") and len(up.solve_times) > 0:
            avg_time_ms = 1e3 * np.mean(up.solve_times)
        else:
            num_solves = max(1, int(np.ceil(4000 / 5.0)))
            avg_time_ms = 1e3 * (end - start) / num_solves

        RMS_err, _ = logMetric(log)

        avg_times_ms.append(avg_time_ms)
        rms_errors.append(RMS_err)

        print(f"  avg time: {avg_time_ms:.4f} ms,  RMS err: {RMS_err:.4f}")

    avg_times_ms = np.array(avg_times_ms)
    rms_errors   = np.array(rms_errors)

    # ★ Save everything to disk
    np.savez(
        DATA_FILE,
        vals=vals,
        avg_times_ms=avg_times_ms,
        rms_errors=rms_errors,
    )
    print(f"\nSaved results to {DATA_FILE}")

# sweep code
# if __name__ == "__main__":
#     # Option A: only recompute if data file is missing
#     DATA_FILE = "osqp_sweep_results.npz"
#     if not os.path.exists(DATA_FILE):
#         run_sweep_and_save()
#
#     DATA_FILE_QPOAS = "qpoases_sweep_results.npz"
#     if not os.path.exists(DATA_FILE_QPOAS):
#         run_sweep_and_save_non_osqp(DATA_FILE_QPOAS)


    # Option B: force recompute
    # run_sweep_and_save()

# if __name__ == "__main__":
    # plot_from_file()
#     # Sweep over OSQP tolerances
#     vals = np.logspace(-5, 0, 50)   # 50 log-spaced tolerances from 1e-5 to 1
#
#     avg_times_ms = []
#     rms_errors   = []
#
#     for err_tol in vals:
#         print(f"\n=== Running trial with eps_rel = eps_abs = {err_tol:.2e} ===")
#
#         # Build MPC with this tolerance
#         up, upc = createMPC(
#             use_QPOases=False,
#             eps_rel=err_tol,
#             eps_abs=err_tol,
#         )
#
#         # Helix task (same as your example, but no plotting)
#         start = time.time()
#         log = controlTest(
#             up,
#             tend=4000,       # same horizon as your helix example
#             useMPC=True,
#             trajAmp=100,     # radius of helix in mm
#             trajFreq=1,      # 1 Hz lateral motion
#             hlInterval=5,
#             tpert=2000,
#             showPlots=False  # IMPORTANT: keep off for batch runs
#         )
#         end = time.time()
#
#         # ---- Average OSQP time per MPC solve ----
#         # Prefer the OSQP-reported times if available
#         if hasattr(up, "solve_times") and len(up.solve_times) > 0:
#             avg_time_ms = 1e3 * np.mean(up.solve_times)
#         else:
#             # Fallback: approximate from wall-clock time
#             # hlInterval=5 ms, tend=4000 ms => ~ tend/hlInterval solves
#             num_solves = max(1, int(np.ceil(4000 / 5.0)))
#             avg_time_ms = 1e3 * (end - start) / num_solves
#
#         avg_times_ms.append(avg_time_ms)
#
#         # ---- RMS tracking error ----
#         RMS_err, _ = logMetric(log)
#         rms_errors.append(RMS_err)
#
#         print(f"Average OSQP time: {avg_time_ms:.4f} ms")
#         print(f"RMS tracking error: {RMS_err:.4f} (in mm, from logMetric)")
#
#     # Convert to numpy arrays for convenience
#     avg_times_ms = np.array(avg_times_ms)
#     rms_errors   = np.array(rms_errors)
#
#     # ---- Plot: average OSQP time vs. tolerance ----
#     plt.figure()
#     plt.semilogx(vals, avg_times_ms, marker='o')
#     plt.xlabel(r"OSQP tolerance  $\epsilon$  (eps\_rel = eps\_abs)")
#     plt.ylabel("Average OSQP solve time [ms]")
#     plt.title("Average OSQP solve time vs. tolerance")
#     plt.grid(True, which='both')
#     plt.tight_layout()
#
#     # ---- Plot: RMS tracking error vs. tolerance ----
#     plt.figure()
#     plt.semilogx(vals, rms_errors, marker='o')
#     plt.xlabel(r"OSQP tolerance  $\epsilon$  (eps\_rel = eps\_abs)")
#     plt.ylabel("RMS tracking error [mm]")
#     plt.title("RMS tracking error vs. tolerance")
#     plt.grid(True, which='both')
#     plt.tight_layout()
#
#     plt.show()
#
#     plt.figure()
#     plt.semilogx(vals, rms_errors, marker='o')
#     plt.xlabel(r"OSQP tolerance  $\epsilon$  (eps\_rel = eps\_abs)")
#     plt.ylabel("RMS tracking error [mm]")
#     plt.title("RMS tracking error vs. tolerance")
#     plt.grid(True, which='both')
#     plt.tight_layout()
#     plt.xlim(1e-3,1)
#
#     plt.show()

#
if __name__ == "__main__":
    up, upc = createMPC(solver="OSQP",eps_abs=1e-4)

    # Hover
    start = time.time()
    # controlTest(upc, 500, useMPC=True, hlInterval=5)

    # Some interesting things you might want to plot
    # - Desired trajectory as a dotted line. When viewControlTestLog, set desTraj to true.

    ## helix task
    simEndTime = 5000
    log = controlTest(
        up,
        tend=simEndTime,  # longer sim so you see the helix
        useMPC=True,
        trajAmp=100,  # radius of helix in mm
        trajFreq=1,  # 1 Hz lateral motion
        hlInterval=5,
        tpert = 3000, #
        showPlots = True
    )
    viewControlTestLog(log,vscale=75,label_fontsize=20,tick_fontsize=15,desTraj=True)

    # show z trajectory
    plt.figure()
    plt.plot(1e-3 * log['t'], log['y'][:, 2], label='z (actual)')  # [mm]

    # optional: desired x trajectory
    plt.plot(1e-3 * log['t'], log['pdes'][:, 2], 'k--', alpha=0.5, label='z (desired)')
    label_fontsize = 20  # axis labels
    tick_fontsize = 18  # tick labels

    plt.xlabel('t [s]',fontsize=label_fontsize)
    plt.ylabel('z [mm]',fontsize=label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.grid(True)
    plt.legend(fontsize=0.75*tick_fontsize)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(1e-3 * log['t'], log['y'][:, 0], label='x (actual)')  # [mm]

    # optional: desired x trajectory
    plt.plot(1e-3 * log['t'], log['pdes'][:, 0], 'k--', alpha=0.5, label='x (desired)')
    label_fontsize = 20  # axis labels
    tick_fontsize = 18  # tick labels

    plt.xlabel('t [s]',fontsize=label_fontsize)
    plt.ylabel('x [mm]',fontsize=label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.grid(True)
    plt.legend(fontsize=0.75*tick_fontsize)
    plt.tight_layout()
    plt.show()
#
    # # flip task
    # log = controlTest(
    #     up,
    #     tend=1000,
    #     useMPC=True,
    #     flipTask=True,
    #     showPlots=False   # or False if you’ll use viewControlTestLog yourself
    # )
    # viewControlTestLog(log,vscale=25,desTraj=True)

#     # perch task
#     # log = controlTest(
#     #     up,
#     #     tend=1000,
#     #     useMPC=True,
#     #     perchTraj=True,
#     #     showPlots=False,
#     # )
#     # viewControlTestLog(log,vscale=25,desTraj=True)
#
#     end = time.time()
#     print("Total sim time (s):", end - start)
#
#     # ---- Plot OSQP timing per MPC call ----
    if hasattr(up, "solve_times") and len(up.solve_times) > 0:
        # Font / figure size parameters
        num_solves = len(up.solve_times)

        t_mpc_ms = np.linspace(0,simEndTime,num_solves,endpoint=False)
        label_fontsize = 18  # axis labels
        tick_fontsize = 16  # tick labels
        title_fontsize = 16  # title
        fig_size = (6, 4)  # inches (width, height)

        plt.figure(figsize=fig_size)
        plt.plot(t_mpc_ms/1000,up.solve_times, marker='o', linestyle='-')

        plt.xlabel("Simulation time [s]", fontsize=label_fontsize)
        plt.ylabel("OSQP solve time [ms]", fontsize=label_fontsize)
        # plt.title("OSQP solve time per MPC call", fontsize=title_fontsize)

        # Set tick label size for both axes
        plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        plt.grid(True)
        plt.tight_layout()
        # plt.ylim(0, 0.1)
        plt.show()



#
#         # Optional: also plot iteration counts
#         # plt.figure()
#         # plt.plot(up.solve_iters, marker='o', linestyle='-')
#         # plt.xlabel("MPC solve index")
#         # plt.ylabel("OSQP iterations")
#         # plt.title("OSQP iterations per MPC call")
#         # plt.grid(True)
#         # plt.tight_layout()
#         # plt.show()
#     RMS_err,_ = logMetric(log)
#     print(RMS_err)




