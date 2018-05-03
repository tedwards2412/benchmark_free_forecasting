from __future__ import division
import h5py
import pylab as plt
import numpy as np
import swordfish as sf
from tqdm import tqdm
from sklearn import svm
from scipy.linalg import eig, eigvals
import WIMpy.DMUtils as DMU
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

s1 = np.linspace(3,70,num=20)
s1width = s1[1]-s1[0]
s1means = s1[0:-1]+s1width/2.
bkgs = ['acc','Anom','ElectronRecoil','n','Neutrino','Wall']

ER = np.linspace(32,200,num=20) # keV
ER_width = ER[1]-ER[0]
ER_c = ER[0:-1] + ER_width/2

def dRdS1(S1, m_DM, epsilon=1e-5, mu_x=1e-5, millicharge=False, m_dipole=False, Nevents=False,  **kwargs):

    eff1, eff2 = np.loadtxt("../Swordfish_Xenon1T/Efficiency-1705.06655.txt", unpack=True)
    efficiency = UnivariateSpline(eff1, eff2, ext="zeros", k=1, s=0)
    S1_vals, E_vals = np.loadtxt("../Swordfish_Xenon1T/S1vsER.txt", unpack=True)
    CalcER = UnivariateSpline(S1_vals, E_vals, k=4, s=0)
    dERdS1 = CalcER.derivative()

    ER_keV = CalcER(S1)
    prefactor = 0.475*efficiency(ER_keV)

    def dRdE(ER_keV, m_x, **kwargs):
        #Load in the list of nuclear spins, atomic masses and mass fractions
        nuclei_Xe = ["Xe128", "Xe129", "Xe130", "Xe131", "Xe132", "Xe134", "Xe136"]
        nuclei_list = np.loadtxt("Nuclei.txt", usecols=(0,), dtype='string')
        frac_list = np.loadtxt("Nuclei.txt", usecols=(3,))
        frac_vals = dict(zip(nuclei_list, frac_list))
        
        dRdE = np.zeros_like(ER_keV)
        for nuc in nuclei_Xe:
            if millicharge:
                dRdE += frac_vals[nuc]*DMU.dRdE_millicharge(ER_keV, m_x, epsilon, nuc, **kwargs)
            if m_dipole:
                dRdE += frac_vals[nuc]*DMU.dRdE_magnetic(ER_keV, m_x, mu_x, nuc, **kwargs)
        return dRdE
        
    dRdEXe = dRdE(ER_keV, m_DM, **kwargs) 
    signal = prefactor*dRdEXe*dERdS1(S1)
    if Nevents:
        return signal, sum(signal*s1width)
    else:
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


ER = np.linspace(32,200,num=20) # keV
ER_width = ER[1]-ER[0]
ER_c = ER[0:-1] + ER_width/2

def load_efficiency():
    eff1, eff2 = np.loadtxt("../DD_files/NR_Darkside50.txt", unpack=True)
    eff1_extra = np.linspace(eff1[-1],205,num=50)
    eff2_extra = np.zeros_like(eff1_extra) + 0.9
    eff1 = np.append(eff1,eff1_extra)
    eff2 = np.append(eff2/100.,eff2_extra)
    return interp1d(eff1,eff2)

def dRdEAr(m, E, eff, epsilon=1e-5, mu_x=1e-5, millicharge=False, m_dipole=False, **kwargs):
    s = np.zeros_like(E)
    if millicharge:
        s += eff(E)*DMU.dRdE_millicharge(E, m, epsilon, "Ar40", **kwargs)
    if m_dipole:
        s += eff(E)*DMU.dRdE_magnetic(E, m, mu_x, "Ar40", **kwargs)
    return s + 1e-30

def Calclims():
    # We are now working in distributions as a function of s1
    mlist = np.logspace(1, 3.9, 100) # GeV
    ULlist_Xenon100T = []
    Bdipolenorm = 1e-10

    b_dict = load_bkgs()
    obsT = np.ones_like(s1means)*35636.
    obsT2 = np.ones_like(s1means)*35636.*100
    b = np.array(b_dict[bkgs[0]]/obsT)
    B = [b_dict[bkgs[0]]/obsT, b_dict[bkgs[1]]/obsT, b_dict[bkgs[2]]/obsT,
        b_dict[bkgs[3]]/obsT, b_dict[bkgs[4]]/obsT, b_dict[bkgs[5]]/obsT]

    SF = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT)
    SF2 = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT2)

    ULlist_Xenon1T_milli = []
    ULlist_Xenon100T_milli = []

    for m in mlist:
        dRdS = np.array(dRdS1(s1means, m, mu_x=Bdipolenorm, m_dipole=True)*s1width)
        UL1T = SF.upperlimit(dRdS, 0.1)
        UL100T = SF2.upperlimit(dRdS, 0.1)
        ULlist_Xenon1T_milli.append(UL1T*Bdipolenorm**2)
        ULlist_Xenon100T_milli.append(UL100T*Bdipolenorm**2)

    ULlist_Xenon1T_milli = np.array(np.sqrt(ULlist_Xenon1T_milli))
    ULlist_Xenon100T_milli = np.array(np.sqrt(ULlist_Xenon100T_milli))
    return interp1d(mlist, ULlist_Xenon1T_milli ), interp1d(mlist, ULlist_Xenon100T_milli)

def Euclideanizemilli(nsamples = 100000):
    Limit1, Limit2 = Calclims()
    b_dict = load_bkgs()
    obsT = np.ones_like(s1means)*35636.*100
    b = np.array(b_dict[bkgs[0]]/obsT)
    B = [b_dict[bkgs[0]]/obsT, b_dict[bkgs[1]]/obsT, b_dict[bkgs[2]]/obsT,
        b_dict[bkgs[3]]/obsT, b_dict[bkgs[4]]/obsT, b_dict[bkgs[5]]/obsT]

    couplings = []
    ESXe = []
    ESAr = []
    NXe = [] 
    NAr = []
    NuisanceES = []
    ms = np.logspace(1,3.9,nsamples/1000)

    sigmav_mean=156.0 
    err_sigv = 13.
    vesc_mean=533.0
    err_vesc = 54.
    vlag_mean=242.0
    err_vlag = 10.

    from random import randint
    sigmav = np.random.uniform(sigmav_mean-(2.*err_sigv),sigmav_mean+(2.*err_sigv),int(nsamples))
    vesc = np.random.uniform(vesc_mean-(2.*err_vesc),vesc_mean+(2.*err_vesc),int(nsamples))
    vlag = np.random.uniform(vlag_mean-(2.*err_vlag),vlag_mean+(2.*err_vlag),int(nsamples))
    
    random_points = np.unique([randint(0, vlag.shape[0]-1) for _ in range(int(nsamples*0.2))])
    sigmav[random_points] = sigmav_mean
    vesc[random_points] = vesc_mean
    vlag[random_points] = vlag_mean

    for m in tqdm(ms, desc="Euclideanizing Xenon-nT"):
        i = 0
        coupling_temp = np.logspace(np.log10(Limit1(m)),np.log10(Limit2(m)), nsamples/100)
        for mu_x in coupling_temp:
            couplings.append([m,0.,mu_x])
            dRdS, Nsig = np.array(dRdS1(s1means, m, mu_x=mu_x, m_dipole=True, Nevents=True, sigmav=sigmav[i], vesc=vesc[i], vlag=vlag[i]))

            SF = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT)
            NXe.append(Nsig*obsT[0])
            ES_temp = SF.euclideanizedsignal(dRdS*s1width)

            NuisanceES_temp = [(sigmav[i]-sigmav_mean)/err_sigv]
            NuisanceES_temp.append((vesc[i]-vesc_mean)/err_vesc)
            NuisanceES_temp.append((vlag[i]-vlag_mean)/err_vlag)
            NuisanceES.append(NuisanceES_temp)
            ESXe.append(ES_temp)
            i += 1

    couplings = np.array(couplings)

    obsT = np.ones_like(ER_c)*1422.
    obsT2 = np.ones_like(ER_c)*1422.*100.*50.
    b = np.zeros_like(ER_c) + 0.1/obsT2/len(ER_c)
    SFAr = sf.Swordfish([b], T=[0.01], E=obsT2)
    eff_temp = load_efficiency()

    for i, m in enumerate(tqdm(couplings[:,0], desc="Euclideanizing DS20k")):
            m = couplings[i,0]
            mu_x = couplings[i,2]

            dRdE = dRdEAr(m, ER_c, eff_temp, mu_x=mu_x, m_dipole=True, sigmav=sigmav[i], vesc=vesc[i], vlag=vlag[i])
            ES_temp = SFAr.euclideanizedsignal(dRdE*ER_width)
            ESAr.append(ES_temp)

#######################

    ESXe = np.array(ESXe)
    ESAr = np.array(ESAr)
    NXe = np.array(NXe)
    NuisanceES = np.array(NuisanceES)

    # Output to new hdf5
    outfile = "../hdf5/Xenon100T_DS20k_gridscanBdipole_HaloTrue.hdf5"
    hf = h5py.File(outfile, 'w')
    hf.create_dataset('ESAr', data=ESAr)
    hf.create_dataset('ESXe', data=ESXe)
    hf.create_dataset('NuisanceES', data=NuisanceES)
    hf.create_dataset('c', data=couplings)
    hf.create_dataset('NXe', data=NXe)
    # hf.create_dataset('NAr', data=NAr)
    hf.close()
    return None

Euclideanizemilli(nsamples = 100000)