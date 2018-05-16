from __future__ import division
import swordfish as sf
import pylab as plt
from matplotlib_venn import venn3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rc
from WIMpy import DMUtils as DMU
from scipy.interpolate import UnivariateSpline
from scipy.linalg import eig, eigvals
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree
from sklearn import svm
from random import randint
from itertools import cycle
import h5py

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

########################## Above is just for signal etc

def operatordiscrim(operator=11, both=False):
    root01 = h5py.File('../hdf5/Xenon100T_DS20k_gridscan01_HaloTrue.hdf5')
    couplings01 = np.array(root01['c'])
    random_points = np.unique([randint(0, couplings01.shape[0]-1) for _ in range(5000)])

    ES01Xe = np.array(root01['ESXe'])
    ES01Ar = np.array(root01['ESAr'])
    NuisanceES = np.array(root01['NuisanceES'])
    couplings01 = couplings01
    mass01 = np.array(root01['mass'])

    c01 = np.zeros([couplings01.shape[0], couplings01.shape[1]+1])
    c01[:,0] = mass01
    c01[:,1:] = couplings01

    ##################

    root011 = h5py.File('../hdf5/Xenon100T_DS20k_gridscan0'+str(operator)+'_HaloTrue.hdf5')
    couplings011 = np.array(root011['c'])

    ES011Xe = np.array(root011['ESXe'])
    ES011Ar = np.array(root011['ESAr'])
    NuisanceES11 = np.array(root011['NuisanceES'])
    mass011 = np.array(root011['mass'])

    c011 = np.zeros([couplings011.shape[0], couplings011.shape[1]+1])
    c011[:,0] = mass011
    c011[:,1:] = couplings011

    ESTmp = np.append(ES01Xe, ES01Ar, axis=1)
    ESHaloUnXe = np.append(ES01Xe, NuisanceES, axis=1)
    ESHaloUnXeAr = np.append(ESTmp, NuisanceES, axis=1)
    
    ESTmp11 = np.append(ES011Xe, ES011Ar, axis=1)
    ESHaloUnXe11 = np.append(ES011Xe, NuisanceES11, axis=1)
    ESHaloUnXeAr11 = np.append(ESTmp11, NuisanceES11, axis=1)

    cXe = np.vstack((c01, c011))
    ESXe = np.vstack((ESHaloUnXe, ESHaloUnXe11))

    cXeAr = np.vstack((c01, c011))
    ESXeAr = np.vstack((ESHaloUnXeAr, ESHaloUnXeAr11))


    shXe = sf.SignalHandler(cXe, ESXe)
    shXeAr = sf.SignalHandler(cXeAr, ESXeAr)


    discrimination = []

    for i in tqdm(range(len(c01[:,0]))):
        P0 = c01[i,:]
        pp, el_ind = shXe.query_region(P0, 2.0, return_indices = True)
        if sum(pp[:,operator] > 0.0) > 0:
            discrimination.append(0.)
        else:
            discrimination.append(1.)

    for i in tqdm(range(len(discrimination))):
        P0 = c01[i,:]
        pp, el_ind = shXeAr.query_region(P0, 2.0, return_indices = True)
        if sum(pp[:,operator] > 0.0) > 0:
            discrimination[i] += 0.
        else:
            discrimination[i] += 1.

    # print discrimination,c01[:,0], c01[:,1]
    from scipy.stats import binned_statistic
    discrimination = np.array(discrimination)
    percentile = lambda x: np.percentile(x, 5)
    if both:
        line, bins, _ = binned_statistic(c01[discrimination==2,0], c01[discrimination==2,1], percentile, bins=np.logspace(1,4,num=40))
        bin_c = bins[:-1] + np.diff(bins)
        mp = 0.938 # GeV
        mu = bin_c*mp/(bin_c + mp)
        linesig = (line)**2 * (mu**2/np.pi) * (1.98e-14**2)
        plt.plot(bin_c, linesig, ls=next(linecycler), label="O" + str(operator) + " - Xe")

    line, bins, _ = binned_statistic(c01[np.logical_or(discrimination==1,discrimination==2),0], c01
    [np.logical_or(discrimination==1,discrimination==2),1], percentile, bins=np.logspace(1,4,num=40))
    bin_c = bins[:-1] + np.diff(bins)

    bin_c = bins[:-1] + np.diff(bins)
    mp = 0.938 # GeV
    mu = bin_c*mp/(bin_c + mp)
    linesig = (line)**2 * (mu**2/np.pi) * (1.98e-14**2)
    plt.plot(bin_c, linesig, ls=next(linecycler), label="O" + str(operator) + " - Xe + Ar")
    return None

def modeldiscrim(limit, millicharge = True):
    root01 = h5py.File('../hdf5/Xenon100T_DS20k_gridscan01_HaloTrue.hdf5')
    couplings01 = np.array(root01['c'])
    random_points = np.unique([randint(0, couplings01.shape[0]-1) for _ in range(1000)])

    ES01Xe = np.array(root01['ESXe'])
    ES01Ar = np.array(root01['ESAr'])
    NuisanceES = np.array(root01['NuisanceES'])
    couplings01 = couplings01
    mass01 = np.array(root01['mass'])

    c01 = np.zeros([couplings01.shape[0], couplings01.shape[1]+1])
    c01[:,0] = mass01
    c01[:,1:] = couplings01

    ##################

    if millicharge:
        rootmodel = h5py.File('../hdf5/Xenon100T_DS20k_gridscanmillicharge_HaloTrue.hdf5')
    else:
        rootmodel = h5py.File('../hdf5/Xenon100T_DS20k_gridscanBdipole_HaloTrue.hdf5')

    cmodel = np.array(rootmodel['c'])
    cmodel_01 = np.zeros([cmodel.shape[0], couplings01.shape[1]+1])
    cmodel_01[:,:3] = cmodel
    ESmodelXe = np.array(rootmodel['ESXe'])
    ESmodelAr = np.array(rootmodel['ESAr'])
    NuisanceESmodel = np.array(rootmodel['NuisanceES'])

    ESTmp = np.append(ES01Xe, ES01Ar, axis=1)
    ESHaloUnXe = np.append(ES01Xe, NuisanceES, axis=1)
    ESHaloUnXeAr = np.append(ESTmp, NuisanceES, axis=1)
    
    ESTmpmodel = np.append(ESmodelXe, ESmodelAr, axis=1)
    ESHaloUnXemodel = np.append(ESmodelXe, NuisanceESmodel, axis=1)
    ESHaloUnXeArmodel = np.append(ESTmpmodel, NuisanceESmodel, axis=1)

    cXe = np.vstack((c01, cmodel_01))
    ESXe = np.vstack((ESHaloUnXe, ESHaloUnXemodel))

    cXeAr = np.vstack((c01, cmodel_01))
    ESXeAr = np.vstack((ESHaloUnXeAr, ESHaloUnXeArmodel))

    shXe = sf.SignalHandler(cXe, ESXe)
    shXeAr = sf.SignalHandler(cXeAr, ESXeAr)

    discrimination = []
    if millicharge:
        a = 1
    else:
        a = 2

    for i in tqdm(range(len(c01[:,0]))):
        P0 = c01[i,:]
        pp, el_ind = shXe.query_region(P0, 2.0, return_indices = True)
        if sum(pp[:,a] > 0.0) > 0:
            discrimination.append(0.)
        else:
            discrimination.append(1.)

    for i in tqdm(range(len(discrimination))):
        P0 = c01[i,:]
        pp, el_ind = shXeAr.query_region(P0, 2.0, return_indices = True)
        if sum(pp[:,a] > 0.0) > 0:
            discrimination[i] += 0.
        else:
            discrimination[i] += 1.

    from scipy.stats import binned_statistic
    discrimination = np.array(discrimination)
    percentile = lambda x: np.percentile(x, 10)
    line, bins, _ = binned_statistic(c01[np.logical_or(discrimination==1,discrimination==2),0], c01
    [np.logical_or(discrimination==1,discrimination==2),1], percentile, bins=np.logspace(1.,4,num=40))

    bin_c = bins[:-1] + np.diff(bins)
    mp = 0.938 # GeV
    mu = bin_c*mp/(bin_c + mp)
    linesig = (line)**2 * (mu**2/np.pi) * (1.98e-14**2)

    if millicharge:
        plt.plot(bin_c, linesig, ls=next(linecycler), label="Millicharge - Xe + Ar")
    else:
        linesig[linesig>limit(bin_c)] = 0.0
        plt.plot(bin_c, linesig, ls=next(linecycler), label="Magnetic Dipole - Xe + Ar")

    return None


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
    return interp1d(mlist, sig_SI_X1T)

plt.figure(figsize=(5,4))
interp = plt_lims()
# modeldiscrim(millicharge = True)
operatordiscrim(operator=11, both=True)
operatordiscrim(operator=4)
plt.xlabel(r'$m_{\chi}[\mathrm{GeV}]$')
plt.xlim(10,10**(3.9))
plt.ylim(1e-48,1e-42)
plt.text(40, 1.5e-48, r"Regions in which discrimination against $\mathcal{O}$1 is possible")

plt.ylabel(r'$\sigma_{\mathrm{SI}} [\mathrm{cm}^2]$')
modeldiscrim(interp, millicharge = False)
plt.legend(frameon=False, prop={'size': 9},  loc="upper center", bbox_to_anchor=(0.25,1.0))
plt.tight_layout(pad=0.3)
plt.savefig("../plots/O1vsallX1T_XnT_testing.pdf")
