import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from sklearn.preprocessing import StandardScaler
from matplotlib import rc


#import matplotlib
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)


rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'],size=22)
# plt.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm}']

files = [r"..\TrainingData\20g_L.txt", #5
         r"..\TrainingData\2g_L.txt", #16
         r"..\TrainingData\2e_1g_L.txt", #22
         r"..\TrainingData\2e_2g_L.txt", #
         r"..\TrainingData\2e_3g_L.txt", #
         r"..\TrainingData\FEM20g_L.txt"] #5

fig = plt.figure()
ax = fig.add_subplot(111)



colours = ['tab:green','tab:orange','tab:blue','tab:red','k']
labels = ['2 g/L','0.2 g/L','0.02 g/L','0.002 g/L','FEM Model']
linestyles = ['solid','solid','solid','solid','dashed']
linewidths = [2,2,2,2,1]


i=0
for FileName, colour, label, linestyle, linewidth in zip(files, colours,labels,linestyles,linewidths):
    data=np.loadtxt(FileName)
    ax.plot(data[:,0], data[:,1], c=colour, linewidth=linewidth, label=label, linestyle=linestyle)

ax.legend(fontsize = 18)

files = [r"../TrainingData/20g_L.txt", #5
          r"../TrainingData/2g_L.txt", #16
          r"../TrainingData/2e_1g_L.txt", #22
          r"../TrainingData/2e_2g_L.txt", #
          r"../TrainingData/2e_3g_L.txt", #
          r"../TrainingData/FEM20g_L.txt"] #5

for FileName, colour, label, linestyle in zip(files, colours,labels,linestyles):
    data=np.loadtxt(FileName)
    ax.plot(data[:,0], data[:,1], c='k', linewidth=1, linestyle='--')

ax.set_ylabel('Fractional Effective Optical \n Thickness Change ' + r'($\frac{EOT_t - EOT_0}{EOT_0}$)', fontsize = 28)

#ax.set_ylabel('Reflectance (a.u.)\n', fontsize = 'large')
ax.set_xlabel('Time (s)', fontsize = 28)
#plt.tick_params(axis='y', left = False, labelleft = False)

#ax.set_xlim([420,900])
#ax.set_ylim([0,50])

# ax.text(445, 10, '(d)', fontsize = 'large')


# ax.text(765, 6.8, r'$55$ $mA \cdot cm^{-2}$', fontsize = 24, color = 'k')
# ax.text(765, 2.4, r'$40$ $mA \cdot cm^{-2}$', fontsize = 24, color = 'k')
# ax.text(765, -2, r'$25$ $mA \cdot cm^{-2}$', fontsize = 24, color = 'k')


fig.set_figwidth(6*1.5)
fig.set_figheight(4.8*1.5)

plt.savefig("..\Figures\FEMExperimentalFitExamples.tif", dpi=200, bbox_inches='tight')