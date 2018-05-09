from __future__ import division
import h5py
import pylab as plt
import numpy as np
import swordfish as sf
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree
from sklearn import svm
from scipy.linalg import eig, eigvals
import WIMpy.DMUtils as DMU
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import interpolate

eff1, eff2 = np.loadtxt("../Swordfish_Xenon1T/Efficiency-1705.06655.txt", unpack=True)
efficiency = UnivariateSpline(eff1, eff2, ext="zeros", k=1, s=0)
S1_vals, E_vals = np.loadtxt("../Swordfish_Xenon1T/S1vsER.txt", unpack=True)

# Interpolation for the recoil energy as a function of S1
# and the derivative
CalcER = UnivariateSpline(S1_vals, E_vals, k=4, s=0)
dERdS1 = CalcER.derivative()

# Recoil distribution as a function of S1
# taking into account the efficiency and change
# of variables ER -> S1
def dRdS1(S1, m_DM, cp_random, cn_random, cov=False):
    # Will also output signal covariance matrix
    # Factor of 0.475 comes from the fact that
    # the reference region should contain about
    # 47.5% of nuclear recoils (between median and 2sigma lines)
    # R = DMU.dRdE_NREFT(ER_keV, A_Xe, m_DM, cp_random, cn_random, "Xe131")
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
        
    dRdEXe = dRdE(ER_keV, m_DM, cp_random, cn_random) 
    signal = prefactor*dRdEXe*dERdS1(S1)
    
    return signal

# We are now working in distributions as a function of s1
s1 = np.linspace(3,70,num=20)
s1width = s1[1]-s1[0]
s1means = s1[0:-1]+s1width/2.
bkgs = ['acc','Anom','ElectronRecoil','n','Neutrino','Wall']

def load_bkgs():
    b = dict()
    for i in range(len(bkgs)):
        S1, temp = np.loadtxt("../DD_files/" + bkgs[i] + ".txt", unpack=True)
        interp = interp1d(S1, temp, bounds_error=False, fill_value=0.0)
        b[bkgs[i]] = interp(s1means)
    return b

mlist = np.logspace(1, 3.9, 100) # GeV
b_dict = load_bkgs()
obsT = np.ones_like(s1means)*35636.
obsT2 = np.ones_like(s1means)*35636.*100
B = [b_dict[bkgs[0]]/obsT, b_dict[bkgs[1]]/obsT, b_dict[bkgs[2]]/obsT,
    b_dict[bkgs[3]]/obsT, b_dict[bkgs[4]]/obsT, b_dict[bkgs[5]]/obsT]

SF1 = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT)
SF2 = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT2)


def genO1(Euclideanize=False):
    norm01 = 9.756e-10
    cp = np.zeros([11])
    cp[0] = norm01
    cn = cp
    ULlist = []
    ULlist_Xenon100T = []

    for i, m in enumerate(mlist):
        dRdS = dRdS1(s1means, m, cp, cn)*s1width
        UL1 = SF1.upperlimit(dRdS, 0.1)
        ULlist.append(UL1*norm01**2)

        UL2 = SF2.upperlimit(dRdS, 0.1)
        ULlist_Xenon100T.append(UL2*norm01**2)

    ULlist_O1 = np.array(np.sqrt(ULlist))
    ULlist_O1_Xenon100T = np.array(np.sqrt(ULlist_Xenon100T))

    couplings = []
    mass = []
    ES = []
    ms = np.logspace(1,3.9,100)

    if Euclideanize:
        Xenon1T_interplimO1 = interpolate.interp1d(mlist, ULlist_O1)
        Xenon100T_interplimO1 = interpolate.interp1d(mlist, ULlist_O1_Xenon100T)

        for m in tqdm(ms, desc="Euclideanizing"):
            coupling_temp = np.linspace(Xenon1T_interplimO1(m),Xenon100T_interplimO1(m), 1000)
            for c_temp in coupling_temp:
                cp = np.zeros(11)
                cp[0] = c_temp
                cn = cp
                c = np.append(cp, cn)
                couplings.append(c)
                mass.append(m)
                dRdS = dRdS1(s1means, m, cp, cn)*s1width
                ES.append(SF2.euclideanizedsignal(dRdS))

        ES = np.array(ES)
        couplings = np.array(couplings)
        mass = np.array(mass)

        # Output to new hdf5
        outfile = "../hdf5/Xenon100T_gridscan01_Euclideanized_dRdS1.hdf5"
        hf = h5py.File(outfile, 'w')
        hf.create_dataset('ES', data=ES)
        hf.create_dataset('mass', data=mass)
        hf.create_dataset('c', data=couplings)
        hf.close()
        
        return ULlist_O1, ULlist_O1_Xenon100T
    else:
        return ULlist_O1, ULlist_O1_Xenon100T


def genO11(Euclideanize=False):
    norm011 = 1e-8
    cp11 = np.zeros(11)

    # Note that the operator number does not match the index in the couplings
    # index = operator number - 1

    cp11[10] = norm011
    cn11 = cp11
    ULlist_011 = []
    ULlist_Xenon100T_011 = []

    for i, m in enumerate(mlist):
        dRdS011 = dRdS1(s1means, m, cp11, cn11)*s1width
        UL11 = SF1.upperlimit(dRdS011, 0.1)
        UL11100T = SF2.upperlimit(dRdS011, 0.1)
        
        ULlist_011.append(UL11*norm011**2)
        ULlist_Xenon100T_011.append(UL11100T*norm011**2)

    ULlist_011 = np.array(np.sqrt(ULlist_011))
    ULlist_Xenon100T_011 = np.array(np.sqrt(ULlist_Xenon100T_011))

    if Euclideanize:
        Xenon1T_interplim = interpolate.interp1d(mlist, ULlist_011)
        Xenon100T_interplim = interpolate.interp1d(mlist, ULlist_Xenon100T_011)

        for m in tqdm(ms, desc="Euclideanizing"):
            coupling_temp = np.linspace(Xenon1T_interplim(m),Xenon100T_interplim(m), 1000)
            for c_temp in coupling_temp:
                cp = np.zeros(11)
                cp[10] = c_temp
                cn = cp
                c = np.append(cp, cn)
                couplings.append(c)
                mass.append(m)
                dRdS_temp = dRdS1(s1means, m, cp, cn)
                ES.append(SF1.euclideanizedsignal(dRdS_temp*s1width))

        ES = np.array(ES)
        couplings = np.array(couplings)
        mass = np.array(mass)

        # Output to new hdf5
        outfile = "../hdf5/Xenon100T_gridscan011_Euclideanized_dRdS1.hdf5"
        hf = h5py.File(outfile, 'w')
        hf.create_dataset('ES', data=ES)
        hf.create_dataset('mass', data=mass)
        hf.create_dataset('c', data=couplings)
        hf.close()

        return ULlist_011, ULlist_Xenon100T_011
    else:
        return ULlist_011, ULlist_Xenon100T_011
    

def genO1_O11():
    ULlist_01, ULlist_Xenon100T_01 = genO1()
    ULlist_011, ULlist_Xenon100T_011 = genO11()

    Xenon1T_interplim_01 = interpolate.interp1d(mlist, ULlist_01)
    Xenon100T_interplim_01 = interpolate.interp1d(mlist, ULlist_Xenon100T_01)

    Xenon1T_interplim_011 = interp1d(mlist, ULlist_011)
    Xenon100T_interplim_011 = interp1d(mlist, ULlist_Xenon100T_011)

    couplings = []
    mass = []
    ES = []

    for m in tqdm(mlist, desc="Euclideanizing"):
        c_01_temp = np.linspace(Xenon1T_interplim_01(m), Xenon100T_interplim_01(m), 32)
        c_011_temp = np.linspace(Xenon1T_interplim_011(m), Xenon100T_interplim_011(m), 32)
        for c_01 in c_01_temp:
            for c_011 in c_011_temp:
                cp = np.zeros(11)
                cp[0] = c_01
                cp[10] = c_011
                cn = cp
                ctemp = np.append(cp, cn)
                couplings.append(ctemp)
                mass.append(m)
                dRdS01011 = dRdS1(s1means, m, cp, cn)*s1width
                UL = SF1.upperlimit(dRdS01011, 0.1)
                if UL > 1:
                    ES.append(SF2.euclideanizedsignal(dRdS01011))
                if UL < 1:
                    continue

    ES = np.array(ES)
    couplings = np.array(couplings)
    mass = np.array(mass)

    # Output to new hdf5
    outfile = "../hdf5/Xenon100T_gridscan01011_Euclideanized_dRdS1.hdf5"
    hf = h5py.File(outfile, 'w')
    hf.create_dataset('ES', data=ES)
    hf.create_dataset('mass', data=mass)
    hf.create_dataset('c', data=couplings)
    hf.close()

def main():
    genO1(Euclideanize=True)
    genO11(Euclideanize=True)
    genO1_O11()

if __name__ == "__main__":
    main()
