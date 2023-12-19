import pandas as pd
import matplotlib.pyplot as plt
import pylab
import matplotlib
import math
from matplotlib import rc
import numpy as np

SequenceLength = 250

df = pd.read_csv(r"../TrainingData/ExperimentalTrainingSet.csv",sep=',',header=None,index_col=0)

time = df.iloc[0,:]/3600
dftrain = df.iloc[1:,:]

concentrations =  ["2",    "0.002",    "0.02",    "0.2",    "0.4",    "1",    "4",    "10",    "20",    "0.1",    "0.04",    "40"]
ExperimentNumber = [15,     13,         14,        3,        17,       10,     8,      22,      2,       4,        2,         2]
Labels =          ["2 g/L","0.002 g/L","0.02 g/L","0.2 g/L","0.4 g/L","1 g/L","4 g/L","10 g/L","20 g/L","0.1 g/L","0.04 g/L","40 g/L"]

fig, ax = plt.subplots()
Means = []
Stdevs = []
for concentration,experimentnumber,label in zip(concentrations,ExperimentNumber,Labels):
    dfConcentration = dftrain.loc[concentration,:]
    plt.plot(time,dfConcentration.iloc[experimentnumber,:],label=label)
    

handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [11,8,7,6,0,5,4,3,9,10,2,1]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="right")
plt.xlim(0,18) 
plt.xticks(range(14))
plt.xlabel("Time (hours)")

ax.set_ylabel('Fractional Effective Optical \n Thickness Change ' + r'($\frac{EOT_t - EOT_0}{EOT_0}$)')


plt.savefig("../Figures/RepresentativeExperimentalTrainingDataset.tif", dpi=200, bbox_inches='tight')
plt.show()