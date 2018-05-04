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
        return s, sum(s*s1width)
    else:
        return s


def load_eff():
    eff1, eff2 = np.loadtxt("../DD_files/NR_Darkside50.txt", unpack=True)
    eff1_extra = np.linspace(eff1[-1],205,num=50)
    eff2_extra = np.zeros_like(eff1_extra) + 0.9
    eff1 = np.append(eff1,eff1_extra)
    eff2 = np.append(eff2/100.,eff2_extra)
    return interp1d(eff1,eff2)

def dRdEAr(m, E, c, eff, Nevents=True, **kwargs):
    s = eff(E)*DMU.dRdE_NREFT(E, m, c, c, "Ar40", **kwargs) + 1.e-30
    if Nevents:
        s, sum(s*ER_width)
    else:
        return s

def load_bkgs():
    b = dict()
    for i in range(len(bkgs)):
        S1, temp = np.loadtxt("../DD_files/" + bkgs[i] + ".txt", unpack=True)
        interp = interp1d(S1, temp, bounds_error=False, fill_value=0.0)
        b[bkgs[i]] = interp(s1means)
    return b

def Calclims():
    # We are now working in distributions as a function of s1
    norm04 = 1.e-8
    mlist = np.logspace(1, 3.9, 100) # GeV
    cp = np.zeros([11])
    cp[3] = norm04
    cn = cp
    ULlist = []

    b_dict = load_bkgs()
    obsT = np.ones_like(s1means)*35636.
    obsT2 = np.ones_like(s1means)*35636.*100
    b = np.array(b_dict[bkgs[0]]/obsT)
    B = [b_dict[bkgs[0]]/obsT, b_dict[bkgs[1]]/obsT, b_dict[bkgs[2]]/obsT,
        b_dict[bkgs[3]]/obsT, b_dict[bkgs[4]]/obsT, b_dict[bkgs[5]]/obsT]

    SF = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT)
    SF2 = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT2)

    cp = np.zeros([11])
    cp[3] = norm04
    cn = cp
    ULlist = []
    ULlist_Xenon100T = []

    for i, m in enumerate(mlist):
        dRdS = dRdS1(s1means, m, cp, cn)*s1width
        UL = SF.upperlimit(dRdS, 0.1)
        UL100 = SF2.upperlimit(dRdS, 0.1)
        ULlist_Xenon100T.append(UL100*norm04**2)
        ULlist.append(UL*norm04**2)

    ULlist = np.array(np.sqrt(ULlist))
    ULlist_Xenon100T = np.array(np.sqrt(ULlist_Xenon100T))
    # plt.loglog(mlist, ULlist)
    # plt.loglog(mlist, ULlist_Xenon100T)
    # plt.show()
    return interp1d(mlist, ULlist), interp1d(mlist, ULlist_Xenon100T)

def EuclideanizeO11(nsamples = 100000):
    Limit1, Limit2 = Calclims()
    b_dict = load_bkgs()
    obsT = np.ones_like(s1means)*35636.*100
    b = np.array(b_dict[bkgs[0]]/obsT)
    B = [b_dict[bkgs[0]]/obsT, b_dict[bkgs[1]]/obsT, b_dict[bkgs[2]]/obsT,
        b_dict[bkgs[3]]/obsT, b_dict[bkgs[4]]/obsT, b_dict[bkgs[5]]/obsT]

    couplings = []
    mass = []
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

    for m in tqdm(ms, desc="Euclideanizing Xenon100T"):
        i = 0
        coupling_temp = np.logspace(np.log10(Limit1(m)),np.log10(Limit2(m)), nsamples/100)
        for c_temp in coupling_temp:
            cp = np.zeros(11)
            cp[3] = c_temp
            cn = cp

            c = np.append(cp, cn)
            couplings.append(c)
            mass.append(m)

            dRdS, Nsig = dRdS1(s1means, m, cp, cn, Nevents=True, sigmav=sigmav[i], vesc=vesc[i], vlag=vlag[i])
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
    mass = np.array(mass)

    obsT = np.ones_like(ER_c)*1422.
    obsT2 = np.ones_like(ER_c)*1422.*100.*50.
    b = np.zeros_like(ER_c) + 0.1/obsT2/len(ER_c)
    SFAr = sf.Swordfish([b], T=[0.01], E=obsT2)
    eff_temp = load_eff()

    for i, m in enumerate(tqdm(mass, desc="Euclideanizing DS20k")):
            cp = np.zeros(11)
            cp[3] = couplings[i,3]
            cn = cp

            dRdE = dRdEAr(m, ER_c, cp, eff_temp, Nevents=False, sigmav=sigmav[i], vesc=vesc[i], vlag=vlag[i])
            ES_temp = SFAr.euclideanizedsignal(dRdE*ER_width)

            ESAr.append(ES_temp)

#######################

    ESXe = np.array(ESXe)
    ESAr = np.array(ESAr)
    NXe = np.array(NXe)

    # Output to new hdf5
    outfile = "../hdf5/Xenon100T_DS20k_gridscan04_HaloTrue.hdf5"
    hf = h5py.File(outfile, 'w')
    hf.create_dataset('ESAr', data=ESAr)
    hf.create_dataset('ESXe', data=ESXe)
    hf.create_dataset('NuisanceES', data=NuisanceES)
    hf.create_dataset('mass', data=mass)
    hf.create_dataset('c', data=couplings)
    hf.create_dataset('NXe', data=NXe)
    # hf.create_dataset('NAr', data=NAr)
    hf.close()
    return None

EuclideanizeO11(nsamples = 100000)