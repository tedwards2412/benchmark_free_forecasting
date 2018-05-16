from __future__ import division
import swordfish as sf
import pylab as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from WIMpy import DMUtils as DMU
from matplotlib import rc
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from random import randint
from itertools import cycle
import h5py

mlist = np.logspace(1, 3.9, 50)
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['cmr']})
rc('font',**{'family':'serif','serif':['cmr']})

eff1, eff2 = np.loadtxt("../Swordfish_Xenon1T/Efficiency-1705.06655.txt", unpack=True)
efficiency = UnivariateSpline(eff1, eff2, ext="zeros", k=1, s=0)
S1_vals, E_vals = np.loadtxt("../Swordfish_Xenon1T/S1vsER.txt", unpack=True)

CalcER = UnivariateSpline(S1_vals, E_vals, k=4, s=0)
dERdS1 = CalcER.derivative()

s1 = np.linspace(3,70,num=20)
s1width = s1[1]-s1[0]
s1means = s1[0:-1]+s1width/2.
bkgs = ['acc','Anom','ElectronRecoil','n','Neutrino','Wall']
linestyle = [":","--","-.", (0, (3, 1, 1, 1, 1, 1))]
linecycler = cycle(linestyle)


def load_bkgs():
    b = dict()
    for i in range(len(bkgs)):
        S1, temp = np.loadtxt("../DD_files/" + bkgs[i] + ".txt", unpack=True)
        interp = interp1d(S1, temp, bounds_error=False, fill_value=0.0)
        b[bkgs[i]] = interp(s1means)
    return b


eff1, eff2 = np.loadtxt("../Swordfish_Xenon1T/Efficiency-1705.06655.txt", unpack=True)
efficiency = UnivariateSpline(eff1, eff2, ext="zeros", k=1, s=0)
S1_vals, E_vals = np.loadtxt("../Swordfish_Xenon1T/S1vsER.txt", unpack=True)

CalcER = UnivariateSpline(S1_vals, E_vals, k=4, s=0)
dERdS1 = CalcER.derivative()

def dRdS1(S1, m_DM, cp_random, cn_random, cov=False):
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

def plt_lims():
    mlist = np.logspace(1, 4., 100) # GeV
    ULlist_Xenon100T = []
    norm01 = 9.756e-10
    cp1 = np.zeros(11)
    cp1[0] = norm01
    cn1 = cp1

    b_dict = load_bkgs()
    obsT = np.ones_like(s1means)*35636.
    obsT2 = np.ones_like(s1means)*35636.*100
    b = np.array(b_dict[bkgs[0]]/obsT)
    B = [b_dict[bkgs[0]]/obsT, b_dict[bkgs[1]]/obsT, b_dict[bkgs[2]]/obsT,
        b_dict[bkgs[3]]/obsT, b_dict[bkgs[4]]/obsT, b_dict[bkgs[5]]/obsT]

    SF = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT)
    SF2 = sf.Swordfish(B, T=[0.01,0.01,0.01,0.01,0.01,0.01], E=obsT2)

    ULlist_Xenon1T = []
    ULlist_Xenon100T = []

    for m in mlist:
        dRdS = np.array(dRdS1(s1means, m, cp1, cn1)*s1width)
        UL1T = SF.upperlimit(dRdS, 0.1)
        UL100T = SF2.upperlimit(dRdS, 0.1)
        ULlist_Xenon1T.append(UL1T*norm01**2)
        ULlist_Xenon100T.append(UL100T*norm01**2)

    ULlist_Xenon1T = np.sqrt(np.array(ULlist_Xenon1T))
    ULlist_Xenon100T = np.sqrt(np.array(ULlist_Xenon100T))

    mp = 0.938 # GeV
    mu = mlist*mp/(mlist + mp)

    sig_SI_X1T = (ULlist_Xenon1T)**2 * (mu**2/np.pi) * (1.98e-14**2)
    sig_SI_X100T = (ULlist_Xenon100T)**2 * (mu**2/np.pi) * (1.98e-14**2)

    plt.loglog(mlist, sig_SI_X1T, label=r"UL XENON1T (2017)")
    plt.loglog(mlist, sig_SI_X100T, label=r"UL XENONnT")
    plt.scatter(mlist[np.argmax(ULlist_Xenon1T)],np.max(ULlist_Xenon1T))
    temp = interp1d(mlist, sig_SI_X1T)
    return temp(10**3.9)

import matplotlib.cm as cm
from random import randint
from random import *

def massdiscrim(matching, operator = 1, Uncertainties = True, Xe = False):
    if Xe:
        Uncertainties = False
    mp = 0.938 # GeV
    mu = mlist*mp/(mlist + mp)

    root01 = h5py.File('../hdf5/Xenon100T_DS20k_gridscan0'+str(operator)+'_HaloTrue.hdf5')
    couplings01 = np.array(root01['c'])
    random_points = np.unique([randint(0, couplings01.shape[0]-1) for _ in range(5000)])

    ES01Xe = np.array(root01['ESXe'])
    ES01Ar = np.array(root01['ESAr'])
    NuisanceES = np.array(root01['NuisanceES'])
    noHaloUn = NuisanceES[:,0] == 0.
    couplings01 = couplings01
    mass01 = np.array(root01['mass'])

    mu1 = mass01*mp/(mass01 + mp)
    couplings01[:,0] = (couplings01[:,0])**2 * (mu1**2/np.pi) * (1.98e-14**2)
    couplings01[:,10] = (couplings01[:,10])**2 * (mu1**2/np.pi) * (1.98e-14**2)
    couplings01[:,3] = (couplings01[:,3])**2 * (mu1**2/np.pi) * (1.98e-14**2)

    c01 = np.zeros([couplings01.shape[0], couplings01.shape[1]+1])
    c01[:,0] = mass01

    # mu1 = mass01*mp/(mass01 + mp)
    # for i in range(couplings01.shape[1]):
    #     couplings01[:,i] = (couplings01[:,i])**2 * (mu1**2/np.pi) * (1.98e-14**2)
    c01[:,1:] = couplings01
    ES01XeNoHaloUn = ES01Xe[noHaloUn]
    ES01ArNoHaloUn = ES01Ar[noHaloUn]
    c01NoHaloUn = c01[noHaloUn,:]

    ESXeAr = np.append(ES01XeNoHaloUn, ES01ArNoHaloUn, axis=1)
    ESTmp = np.append(ES01Xe, ES01Ar, axis=1)
    ESHaloUn = np.append(ESTmp, NuisanceES, axis=1)
    # ESHaloUn = ESTmp

    shXe = sf.SignalHandler(c01NoHaloUn, ES01XeNoHaloUn)
    shXeAr = sf.SignalHandler(c01NoHaloUn, ESXeAr)
    shHaloUn = sf.SignalHandler(c01, ESHaloUn)

    sigma_list_Xe = []
    sigma_list_XeAr = []
    sigma_list_HaloUn = []
    m_listXe = []
    m_listXeAr = []
    m_listHaloUn = []

    if Xe:
        for i in tqdm(range(len(c01NoHaloUn[:,0]))):
            P0 = c01NoHaloUn[i,:]
            pp, el_ind = shXe.query_region(P0, sigma=2.0, d=1, return_indices = True)
            if pp.size == 0.0:
                continue
            if np.max(pp[:,0]) == np.max(mass01):
                m_listXe.append(c01NoHaloUn[i,0])
                sigma_list_Xe.append(c01NoHaloUn[i,operator])

    if not Uncertainties:
        for i in tqdm(range(len(c01NoHaloUn[:,0]))):
            P0 = c01NoHaloUn[i,:]
            pp, el_ind = shXeAr.query_region(P0, sigma=2.0, d=1, return_indices = True)
            if pp.size == 0.0:
                continue
            if np.max(pp[:,0]) == np.max(mass01):
                m_listXeAr.append(c01NoHaloUn[i,0])
                sigma_list_XeAr.append(c01NoHaloUn[i,operator])

    for i in tqdm(range(len(c01[:,0]))):
        P0 = c01[i,:]
        pp, el_ind = shHaloUn.query_region(P0, sigma=2.0, d=1, return_indices = True)
        if pp.size == 0.0:
            continue
        if np.max(pp[:,0]) == np.max(mass01):
            m_listHaloUn.append(c01[i,0])
            sigma_list_HaloUn.append(c01[i,operator])

    ##### Filter results

    sigma_list_Xe = np.array(sigma_list_Xe)
    sigma_list_XeAr = np.array(sigma_list_XeAr)
    sigma_list_HaloUn = np.array(sigma_list_HaloUn)

    m_listXe = np.array(m_listXe)
    m_listXeAr = np.array(m_listXeAr)
    m_listHaloUn = np.array(m_listHaloUn)

    # mlist_temp1 = np.unique(m_listXe)
    # mlist_temp2 = np.unique(m_listXeAr) 
    # mlist_temp3 = np.unique(m_listHaloUn) 

    # sigma_discrimXe = np.zeros_like(mlist_temp1)
    # sigma_discrimXeAr = np.zeros_like(mlist_temp2)
    # sigma_discrimHaloUn = np.zeros_like(mlist_temp3)

    percentile = lambda x: np.percentile(x, 99.9)
    if Xe:
        line, bins, _ = binned_statistic(m_listXe, sigma_list_Xe, percentile, bins=np.logspace(1,3.9,num=40))
        bin_c = bins[:-1] + np.diff(bins)
        if not operator == 1:
            scale = matching/np.max(line)
            line  *= scale
        plt.plot(bin_c, line, ls=next(linecycler), label=r"O" + str(operator) + r" - Xe w/o Halo Uncertainties")

    if not Uncertainties:
        line, bins, _ = binned_statistic(m_listXeAr, sigma_list_XeAr, percentile, bins=np.logspace(1,3.9,num=40))
        bin_c = bins[:-1] + np.diff(bins)
        if not operator == 1:
            scale = matching/np.nanmax(line)
            line  *= scale
        plt.plot(bin_c, line, ls=next(linecycler),label=r"O" + str(operator) + r" - Xe + Ar w/o Halo Uncertainties")

    line, bins, _ = binned_statistic(m_listHaloUn, sigma_list_HaloUn, percentile, bins=np.logspace(1,3.9,num=40))
    bin_c = bins[:-1] + np.diff(bins)
    if not operator == 1:
        scale = matching/np.nanmax(line)
        line  *= scale
        plt.plot(bin_c, line, ls=next(linecycler),label=r"O" + str(operator) + r" - Xe + Ar")
    elif operator == 1:
        plt.plot(bin_c, line, ls=next(linecycler),label=r"O" + str(operator) + r" - Xe + Ar")

def massdiscrimmodels(matching, millicharge=True):
    mp = 0.938 # GeV
    mu = mlist*mp/(mlist + mp)
    if millicharge:
        root01 = h5py.File('../hdf5/Xenon100T_DS20k_gridscanmillicharge_HaloTrue.hdf5')
    else:
        root01 = h5py.File('../hdf5/Xenon100T_DS20k_gridscanBdipole_HaloTrue.hdf5')
    couplings01 = np.array(root01['c'])
    random_points = np.unique([randint(0, couplings01.shape[0]-1) for _ in range(5000)])

    ES01Xe = np.array(root01['ESXe'])
    ES01Ar = np.array(root01['ESAr'])
    NuisanceES = np.array(root01['NuisanceES'])
    noHaloUn = NuisanceES[:,0] == 0.
    c01 = couplings01

    mu1 = c01[:,0]*mp/(c01[:,0] + mp)
    c01[:,1] = (c01[:,1])**2 * (mu1**2/np.pi) * (1.98e-14**2)
    c01[:,2] = (c01[:,2])**2 * (mu1**2/np.pi) * (1.98e-14**2)

    ES01XeNoHaloUn = ES01Xe[noHaloUn]
    ES01ArNoHaloUn = ES01Ar[noHaloUn]
    c01NoHaloUn = c01[noHaloUn,:]

    ESXeAr = np.append(ES01XeNoHaloUn, ES01ArNoHaloUn, axis=1)
    ESTmp = np.append(ES01Xe, ES01Ar, axis=1)
    ESHaloUn = np.append(ESTmp, NuisanceES, axis=1)

    shXe = sf.SignalHandler(c01NoHaloUn, ES01XeNoHaloUn)
    shXeAr = sf.SignalHandler(c01NoHaloUn, ESXeAr)
    shHaloUn = sf.SignalHandler(c01, ESHaloUn)

    sigma_list_Xe = []
    sigma_list_XeAr = []
    sigma_list_HaloUn = []
    m_listXe = []
    m_listXeAr = []
    m_listHaloUn = []

    for i in tqdm(range(len(c01[:,0]))):
        P0 = c01[i,:]
        pp, el_ind = shHaloUn.query_region(P0, sigma=2.0, d=1, return_indices = True)
        if pp.size == 0.0:
            continue
        if np.max(pp[:,0]) == np.max(c01[:,0]):
            m_listHaloUn.append(c01[i,0])
            if millicharge:
                sigma_list_HaloUn.append(c01[i,1])
            else:
                sigma_list_HaloUn.append(c01[i,2])

    ##### Filter results

    sigma_list_Xe = np.array(sigma_list_Xe)
    sigma_list_XeAr = np.array(sigma_list_XeAr)
    sigma_list_HaloUn = np.array(sigma_list_HaloUn)

    m_listXe = np.array(m_listXe)
    m_listXeAr = np.array(m_listXeAr)
    m_listHaloUn = np.array(m_listHaloUn)

    percentile = lambda x: np.percentile(x, 99.9)
    line, bins, _ = binned_statistic(m_listHaloUn, sigma_list_HaloUn, percentile, bins=np.logspace(1,3.9,num=40))
    bin_c = bins[:-1] + np.diff(bins)
    scale = matching/np.nanmax(line)
    line  *= scale

    if millicharge:
        plt.loglog(bin_c, line, ls=next(linecycler), label=r"Millicharge - Xe + Ar")
    elif not millicharge:
        plt.loglog(bin_c, line, ls=next(linecycler), label=r"Magnetic Dipole - Xe + Ar")


plt.figure(figsize=(5,4))
match = plt_lims()
massdiscrim(match, operator = 1, Uncertainties = True, Xe = True)
massdiscrim(match, operator = 11, Uncertainties = True, Xe = False)
massdiscrim(match, operator = 4, Uncertainties = True, Xe = False)
massdiscrimmodels(match, millicharge=True)
massdiscrimmodels(match, millicharge=False)

plt.xlabel(r'$m_{\chi}[\mathrm{GeV}]$')
plt.ylabel(r'$\sigma_{\mathrm{SI}} [\mathrm{cm}^{2}]$')
plt.text(50, 1.5e-48, "Regions in which mass discrimination is not possible")
plt.xlim(10,10**(3.9))
plt.ylim(1e-48,1e-42)
plt.legend(frameon=False, prop={'size': 9}, loc="upper center", bbox_to_anchor=(0.35,1.0))
plt.tight_layout(pad=0.3)
plt.savefig("../plots/mass_discrimination_all_test.pdf")
# massdiscrimmodels(millicharge=False)