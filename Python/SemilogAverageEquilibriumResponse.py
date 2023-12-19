import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
import pylab

rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'],size=12)

#
SequenceLength = 250
markerSize = 80
insetMarkerSize = 40
viridisColour = 70

df = pd.read_csv(r"../TrainingData/ExperimentalTrainingSet.csv",sep=',',header=None,index_col=0)

time = df.iloc[0,1:]/3600
dftrain = df.iloc[1:,:]

concentrations = ["2","0.002","0.02","0.2","0.4","1","4","10","20","0.1","0.04","40"]
fig, ax = plt.subplots()
Means = []
Stdevs = []
for concentration in concentrations:
    Mean = np.mean(dftrain.loc[concentration,SequenceLength])
    Stdev = np.std(dftrain.loc[concentration,SequenceLength],ddof=1)
    Means.append(Mean)
    Stdevs.append(Stdev)
    Concentration = float(concentration)
    plt.scatter(Concentration,Mean, facecolor=(0,0,0,0), edgecolor=pylab.cm.viridis(viridisColour), marker='o', s=markerSize)
    plt.semilogx([Concentration, Concentration], [Mean+Stdev, Mean-Stdev], marker="_", color=pylab.cm.viridis(viridisColour))

Concentrations = [float(i) for i in concentrations]
ResponsevsConcentration = np.array([Concentrations, Means])
ResponsevsConcentration = np.delete(ResponsevsConcentration, [], 1)

def RedlichPetersonIsotherm(concentration, A, B, beta, I):
    return I + A*concentration/(1 + B*(concentration**beta))

popt, pcov = curve_fit(RedlichPetersonIsotherm, ResponsevsConcentration[0,:], ResponsevsConcentration[1,:])

concentrations = np.logspace(np.log(0.002),np.log(40),base=np.e,num=100)
plt.semilogx(concentrations,RedlichPetersonIsotherm(concentrations,popt[0],popt[1],popt[2],popt[3]),color=pylab.cm.viridis(viridisColour))


plt.xlabel('Concentration (mg/mL)')
plt.ylabel('Fractional Effective Optical \n Thickness Change ' + r'($\frac{EOT_t - EOT_0}{EOT_0}$)')
# ax.set_ylabel('Fractional Effective Optical \n Thickness Change ' + r'($\frac{EOT_t - EOT_0}{EOT_0}$)', fontsize = 28)
ax.tick_params(right=True, top=True, labelright=False, labeltop=False)
ax.tick_params(direction="in")

axins = ax.inset_axes([0.15, 0.6, 0.35, 0.35])
axins.scatter(Concentrations,Means, facecolor=(0,0,0,0), edgecolor=pylab.cm.viridis(viridisColour), marker='o', s=insetMarkerSize)
for Concentration,Mean,Stdev in zip(Concentrations,Means,Stdevs):
    axins.plot([Concentration, Concentration], [Mean+Stdev, Mean-Stdev], marker="_", color=pylab.cm.viridis(viridisColour))
axins.plot(concentrations,RedlichPetersonIsotherm(concentrations,popt[0],popt[1],popt[2],popt[3]),color=pylab.cm.viridis(viridisColour))

axins.tick_params(right=True, top=True, labelright=False, labeltop=False)
axins.tick_params(direction="in")
axins.set_xlabel('Concentration (mg/mL)')
axins.set_ylabel(r'($\frac{EOT_t - EOT_0}{EOT_0}$)')

plt.savefig("../Figures/SemilogAverageEquilibriumResponse.tif", dpi=200, bbox_inches='tight')
plt.show()