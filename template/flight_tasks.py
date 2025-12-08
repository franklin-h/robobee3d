"""Tasks specified as desired pos, vel and nominal traj"""
import numpy as np

VERTICAL = np.array([0,0,1])

def helix(t, initialPos, trajAmp=80, trajFreq=1, dz=0.15, useY=True):
    """If trajAmp is small, can be used for hover"""
    posdes = np.copy(initialPos)
    dpdes = np.zeros(3)
    trajOmg = 2 * np.pi * trajFreq * 1e-3 # to KHz, then to rad/ms
    posdes[0] += trajAmp * np.sin(trajOmg * t)
    dpdes[0] = trajAmp * trajOmg * np.cos(trajOmg * t)
    if useY:
        posdes[1] += trajAmp * (1 - np.cos(trajOmg * t))
        dpdes[1] = trajAmp * trajOmg * np.sin(trajOmg * t)
    if trajAmp > 1e-3: # otherwise assume hover
        posdes[2] += dz*t
        dpdes[2] = dz

    return posdes, dpdes, VERTICAL

def straightAcc(t, initialPos, tduration=500, vdes=2):
    """duration in ms, vdes in m/s"""
    pdes = np.copy(initialPos)
    dpdes = np.zeros(3)
    dpdes[0] = vdes if t < tduration else 0
    pdes[0] += vdes * np.clip(t, 0, tduration)
    
    return pdes, dpdes, VERTICAL

def flip(t, initialPos, tstart=100, tend=200, vdes=2):
    pdes = np.copy(initialPos)
    # rotation phase 0 to 1
    ph = np.clip((t - tstart) / tend, 0, 1)
    sdes = np.array([-np.sin(ph*2*np.pi), 0, np.cos(ph*2*np.pi)])
    return pdes, np.zeros(3), sdes

def perch(t, initialPos, tend=500, trotstart=100, trotend=450, vdes=0.2):
    pdes = np.copy(initialPos)
    dpdes = np.zeros(3)
    pdes[0] += vdes * np.clip(t, 0, tend)
    dpdes[0] = vdes if t < tend else 0
    if t < tend:
        # rotation phase 0 to 1
        ph = np.clip((t - trotend) / trotstart, 0, 1)
        sdes = np.array([-np.sin(ph*np.pi), 0, np.cos(ph*np.pi)])
    else:
        sdes = np.array([-1,0,0])
    return pdes, dpdes, sdes

def perch_parab_pet(t, initialPos, tend=500, vdes=0.2):
    pdes = np.zeros((3, len(t)))
    pdes[:, 0] = initialPos
    dpdes = np.zeros((3, len(t)))
    dpdes[:, 0] = vdes

    fpdes = [20, 20, 250]
    sdes = np.zeros((3, len(t)))
    sdes[2, :] = 1


    if initialPos[2] >= fpdes[2]:
        s2 = np.sqrt((fpdes[0] - initialPos[0]) ** 2 + (fpdes[1] - initialPos[1]) ** 2)
        for i in range(1, len(pdes)):
            pdes[i, 0] = pdes[i - 1, 0] + s2 / len(t)
            pdes[i, 1] = pdes[i - 1, 1] + s2 / len(t)
            pdes[i, 2] = pdes[i - 1, 2] + (fpdes[2] - initialPos[2])/len(t)
    else:
        Amp = (fpdes[2] - initialPos[2]) + 10
        s2 = np.sqrt((fpdes[0] - initialPos[0]) ** 2 + (fpdes[1] - initialPos[1]) ** 2)
        D = s2/(1 + np.sqrt(10/Amp))
        for i in range(1, len(pdes)):
            pdes[i, 0] = pdes[i-1, 0] + s2/len(t)
            pdes[i, 1] = pdes[i-1, 1] + s2/len(t)
            pdes[i, 2] = pdes[i-1, 2] + (Amp/D**2)*(np.sqrt((pdes[i, 0] - initialPos[0])**2 + (pdes[i, 1] - initialPos[1])**2) - D**2) + Amp

    return pdes, dpdes, sdes
