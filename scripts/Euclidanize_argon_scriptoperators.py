
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
import h5py

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

def load_eff():
    eff1, eff2 = np.loadtxt("../DD_files/NR_Darkside50.txt", unpack=True)
    eff1_extra = np.linspace(eff1[-1],205,num=50)
    eff2_extra = np.zeros_like(eff1_extra) + 0.9
    eff1 = np.append(eff1,eff1_extra)
    eff2 = np.append(eff2/100.,eff2_extra)
    return interp1d(eff1,eff2)

def dRdE(m, E, c, eff):
    s = eff(E)*DMU.dRdE_NREFT(E, m, c, c, "Ar40") + 1.e-30
    return s

eff_temp = load_eff()


obsT = np.ones_like(ER_c)*1422.
obsT2 = np.ones_like(ER_c)*1422.*100.*50.
#BJK: Changed this from obsT to obsT2
b = np.zeros_like(ER_c) + 0.1/obsT2/len(ER_c)
SF2 = sf.Swordfish(b, T=[0.01], E=obsT2)

#print(obsT2)
#print(b)
##################

root01 = h5py.File('../hdf5/Xenon100T_gridscan01_Euclideanized_dRdS1.hdf5')
couplings01 = np.array(root01['c'])
ES01 = np.array(root01['ES'])
mass01 = np.array(root01['mass'])

ESAr = []

for i in tqdm(range(0,len(mass01)), desc="Euclideanizing"):
    cp = couplings01[i,:11]
    signal = dRdE(mass01[i], ER_c, cp, eff_temp)*ER_width
    ESAr.append(SF2.euclideanizedsignal(signal))

ESAr = np.array(ESAr)
ES = np.append(ES01,ESAr,axis=1)
# Output to new hdf5
outfile = "../hdf5/Xenon_DS_250000_gridscan01_Euclideanized_dRdS1.hdf5"
hf = h5py.File(outfile, 'w')
hf.create_dataset('ES', data=ES)
hf.create_dataset('mass', data=mass01)
hf.create_dataset('c', data=couplings01)
hf.close()

#################

root011 = h5py.File('../hdf5/Xenon100T_gridscan011_Euclideanized_dRdS1.hdf5')
couplings011 = np.array(root011['c'])
ES011 = np.array(root011['ES'])
mass011 = np.array(root011['mass'])

ESAr = []

for i in tqdm(range(0,len(mass011)), desc="Euclideanizing"):
        cp = couplings011[i,:11]
        signal = dRdE(mass011[i], ER_c, cp, eff_temp)*ER_width
        ESAr.append(SF2.euclideanizedsignal(signal))

ESAr = np.array(ESAr)
ES = np.append(ES011,ESAr,axis=1)
# Output to new hdf5
outfile = "../hdf5/Xenon_DS_250000_gridscan011_Euclideanized_dRdS1.hdf5"
hf = h5py.File(outfile, 'w')
hf.create_dataset('ES', data=ES)
hf.create_dataset('mass', data=mass011)
hf.create_dataset('c', data=couplings011)
hf.close()

# ####################

root01011 = h5py.File('../hdf5/Xenon100T_gridscan01011_Euclideanized_dRdS1.hdf5')
couplings01011 = np.array(root01011['c'])
ES01011 = np.array(root01011['ES'])
mass01011 = np.array(root01011['mass'])

ESAr = []

for i in tqdm(range(0,len(mass01011)), desc="Euclideanizing"):
        cp = couplings01011[i,:11]
        signal = dRdE(mass01011[i], ER_c, cp, eff_temp)*ER_width
        ESAr.append(SF2.euclideanizedsignal(signal))

ESAr = np.array(ESAr)
ES = np.append(ES01011,ESAr,axis=1)
# Output to new hdf5
outfile = "../hdf5/Xenon_DS_250000_gridscan01011_Euclideanized_dRdS1.hdf5"
hf = h5py.File(outfile, 'w')
hf.create_dataset('ES', data=ES)
hf.create_dataset('mass', data=mass01011)
hf.create_dataset('c', data=couplings01011)
hf.close()