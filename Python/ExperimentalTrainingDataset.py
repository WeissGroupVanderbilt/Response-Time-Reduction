import pandas as pd
import matplotlib.pyplot as plt
import pylab
import matplotlib
import math
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'],size=12)

df = pd.read_csv(r"../TrainingData/ExperimentalTrainingSet.csv",sep=',',header=None)

time = df.iloc[0,1:]/3600
dftrain = df.iloc[1:,:]

fig, ax = plt.subplots()

for row in dftrain.iterrows():
    colour = round(255*(math.log(float(row[1][0])+1e-2) - math.log(1e-2))/(math.log(40) - math.log(1e-2)))
    plt.plot(time,row[1][1:],c=pylab.cm.viridis(colour))
    
plt.xlabel("Time (hours)")
ax.set_ylabel('Fractional Effective Optical \n Thickness Change ' + r'($\frac{EOT_t - EOT_0}{EOT_0}$)')

sm = plt.cm.ScalarMappable(cmap=pylab.cm.viridis, norm=matplotlib.colors.LogNorm(vmin=0.002, vmax=40))
colorbar = plt.colorbar(sm)
colorbar.ax.get_yaxis().labelpad = 15
colorbar.ax.set_ylabel("Concentration (mg/mL)", rotation=270)
ax.tick_params(right=True, top=True, labelright=False, labeltop=False)
ax.tick_params(direction="in")
plt.savefig(".."+"/Figures/ExperimentalTrainingDataset.tif", dpi=200, bbox_inches='tight')
plt.show()