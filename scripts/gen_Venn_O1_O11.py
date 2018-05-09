from __future__ import division
import h5py
import pylab as plt
import numpy as np
import swordfish as sf
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree
from sklearn import svm
from WIMpy import DMUtils as DMU
from scipy.linalg import eig, eigvals
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

eff1, eff2 = np.loadtxt("../Swordfish_Xenon1T/Efficiency-1705.06655.txt", unpack=True)
efficiency = UnivariateSpline(eff1, eff2, ext="zeros", k=1, s=0)
S1_vals, E_vals = np.loadtxt("../Swordfish_Xenon1T/S1vsER.txt", unpack=True)

# We are now working in distributions as a function of s1
s1 = np.linspace(3,70,num=20)
s1width = s1[1]-s1[0]
s1means = s1[0:-1]+s1width/2.

# Interpolation for the recoil energy as a function of S1
# and the derivative
CalcER = UnivariateSpline(S1_vals, E_vals, k=4, s=0)
dERdS1 = CalcER.derivative()

# Recoil distribution as a function of S1
# taking into account the efficiency and change
# of variables ER -> S1
def dRdS1(S1, m_DM, cp_random, cn_random, Nevents=False, **kwargs):
    eff1, eff2 = np.loadtxt("../Swordfish_Xenon1T/Efficiency-1705.06655.txt", unpack=True)
    efficiency = UnivariateSpline(eff1, eff2, ext="zeros", k=1, s=0)
    S1_vals, E_vals = np.loadtxt("../Swordfish_Xenon1T/S1vsER.txt", unpack=True)
    CalcER = UnivariateSpline(S1_vals, E_vals, k=4, s=0)
    dERdS1 = CalcER.derivative()
    ER_keV = CalcER(S1)
    prefactor = 0.475*efficiency(ER_keV)

    def dRdE(ER_keV, m_x, cp, cn, **kwargs):
        #Load in the list of nuclear spins, atomic masses and mass fractions
        nuclei_Xe = ["Xe128", "Xe129", "Xe130", "Xe131", "Xe132", "Xe134", "Xe136"]
        nuclei_list = np.loadtxt("Nuclei.txt", usecols=(0,), dtype='string')
        frac_list = np.loadtxt("Nuclei.txt", usecols=(3,))
        frac_vals = dict(zip(nuclei_list, frac_list))
        
        dRdE = np.zeros_like(ER_keV)
        for nuc in nuclei_Xe:
            dRdE += frac_vals[nuc]*DMU.dRdE_NREFT(ER_keV, m_x, cp, cn, nuc, **kwargs)
        return dRdE
            
    dRdEXe = dRdE(ER_keV, m_DM, cp_random, cn_random, **kwargs) 
    s = prefactor*dRdEXe*dERdS1(S1)
    
    if Nevents:
        N = sum(s*s1width)
        return s, N
    else:
        return s

def load_bkgs():
    b = dict()
    for i in range(len(bkgs)):
        S1, temp = np.loadtxt("../DD_files/" + bkgs[i] + ".txt", unpack=True)
        interp = interp1d(S1, temp, bounds_error=False, fill_value=0.0)
        b[bkgs[i]] = interp(s1means)
    return b
    
bkgs = ['acc','Anom','ElectronRecoil','n','Neutrino','Wall']
b_dict = load_bkgs()
obsT = np.ones_like(s1means)*35636.
b = np.array(b_dict[bkgs[0]]/obsT)
B = [b_dict[bkgs[0]]/obsT, b_dict[bkgs[1]]/obsT, b_dict[bkgs[2]]/obsT,
    b_dict[bkgs[3]]/obsT, b_dict[bkgs[4]]/obsT, b_dict[bkgs[5]]/obsT]

SF = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT)

def Volumes(Argon = True):
    if Argon:
        filename = '../hdf5/Xenon_DS_250000_'
    else:
        filename = '../hdf5/Xenon100T_'

    R_01 = []
    R_011 = []
    mlist = np.logspace(1,4,100)

    # Calculates the number of signal events in XENONnT for each point

    for m in mlist:
        c_01 = np.zeros(11)
        c_01[0] = 1.
        c_011 = np.zeros(11)
        c_011[10] = 1.
        s1, N1 = dRdS1(s1means, m, c_01, c_01, Nevents=True)
        s11, N11 = dRdS1(s1means, m, c_011, c_011, Nevents=True)
        R_01.append(N1)
        R_011.append(N11)
        
    R_01 = interp1d(mlist, R_01)
    R_011 = interp1d(mlist, R_011)

    # Simply load the files, not using all points since stable results are found with less

    root = h5py.File(filename + 'gridscan01011_Euclideanized_dRdS1.hdf5')
    from random import randint
    couplings01011 = np.array(root['c'])
    random_points1 = np.unique([randint(0, couplings01011.shape[0]-1) for _ in range(50000)])
    ES01011 = np.array(root['ES'])[random_points1]
    mass01011 = np.array(root['mass'])

    c01011 = np.zeros([len(random_points1), couplings01011.shape[1]+1])
    c01011[:,0] = mass01011[random_points1]
    c01011[:,1:] = couplings01011[random_points1]

    ###############

    root01 = h5py.File(filename + 'gridscan01_Euclideanized_dRdS1.hdf5')
    couplings01 = np.array(root01['c'])
    random_points2 = np.unique([randint(0, couplings01.shape[0]-1) for _ in range(50000)])
    ES01 = np.array(root01['ES'])[random_points2]
    mass01 = np.array(root01['mass'])

    c01 = np.zeros([len(random_points2), couplings01.shape[1]+1])
    c01[:,0] = mass01[random_points2]
    c01[:,1:] = couplings01[random_points2]

    ##################

    root011 = h5py.File(filename + 'gridscan011_Euclideanized_dRdS1.hdf5')
    couplings011 = np.array(root011['c'])
    random_points3 = np.unique([randint(0, couplings011.shape[0]-1) for _ in range(50000)])
    ES011 = np.array(root011['ES'])[random_points3]
    mass011 = np.array(root011['mass'])

    c011 = np.zeros([len(random_points3), couplings011.shape[1]+1])
    c011[:,0] = mass011[random_points3]
    c011[:,1:] = couplings011[random_points3]

    c = np.vstack((c01011, c01, c011))
    ES = np.vstack((ES01011, ES01, ES011))

    obsT2 = np.ones_like(s1means)*35636.*100
    Events = (c[:,1]**2*R_01(c[:,0]) + c[:,11]**2*R_011(c[:,0]))*obsT2[0]

    bin_edges = np.linspace(Events.min(), Events.max(),num=8)
    for i in range(bin_edges.size - 1):
        points = np.logical_and(Events > bin_edges[i], Events < bin_edges[i+1])
        print bin_edges[i], bin_edges[i+1]
        print("Number of points in bin = ", sum(points))

    # Generate the Signal Handler object and find masks for the different sections of the Venn diagrams

    sh = sf.SignalHandler(c, ES)

    c01mask = np.zeros(len(c[:,0]), dtype=bool)
    c01mask[c[:,11] == 0.0] = True
    c01mask = sh.shell(c01mask)

    c011mask = np.zeros(len(c[:,0]), dtype=bool)
    c011mask[c[:,1] == 0.0] = True
    c011mask = sh.shell(c011mask)

    cmixmask = np.zeros(len(c[:,0]), dtype=bool)
    cmixmask[np.logical_or(c[:,1] == 0.0, c[:,11] == 0.0)] = True
    cmixmask = sh.shell(cmixmask)

    Volall, wall = sh.volume(d=3, sigma=2., return_individual=True)
    Vol01, w01 = sh.volume(d=3, sigma=2., mask=c01mask, return_individual=True)
    Vol011, w011 = sh.volume(d=3, sigma=2., mask=c011mask, return_individual=True)
    Volmix, wmix = sh.volume(d=3, sigma=2., mask=cmixmask, return_individual=True)

    plotarray = []
    bin_edges = np.linspace(Events.min(), Events.max(),num=25)
    for i in range(bin_edges.size - 1):
        pointsall = np.logical_and(Events > bin_edges[i], Events < bin_edges[i+1])
        points01 = np.logical_and(Events[c01mask] > bin_edges[i], Events[c01mask] < bin_edges[i+1])
        points011 = np.logical_and(Events[c011mask] > bin_edges[i], Events[c011mask] < bin_edges[i+1])
        pointsmix = np.logical_and(Events[cmixmask] > bin_edges[i], Events[cmixmask] < bin_edges[i+1])
        
        Vall = sum(wall[pointsall])
        V01 = sum(w01[points01])
        V011 = sum(w011[points011])
        Vmix = sum(wmix[pointsmix])
        V01and011 = V01+V011-Vmix
        
        l_temp = [bin_edges[i],bin_edges[i+1],V01,V011,Vall,V01and011]
        plotarray.append(l_temp)

    plotarray = np.array(plotarray)
    if Argon:
        np.savetxt("../Notebooks/venn_array_Xe+Ar", plotarray)
    else:
        np.savetxt("venn_array_Xe", plotarray)
    return None

def main():
    Volumes(Argon=True)
    Volumes(Argon=False)

if __name__ == "__main__":
    main()
