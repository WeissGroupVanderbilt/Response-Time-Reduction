import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from LSTMutils import MeanVarianceLogLikelyhoodLoss
from sklearn.model_selection import train_test_split
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'],size=12)

## input parameters

# number of models in the ensemble
NumEnsemble = 10
# length of time series
SequenceLength = 250
validation_split = 0.25
batch_size = 32
TestOrTrainDataset = 'test' #'train' or 'test'
t90Window = 0.1 # t90 is defined as the time until the response settles to within this fraction of the equilibrium response indefinitely
test_split = 0.2

# read experimental data and split into time (s) converted to hours, concentration labels and time series data
df_full = pd.read_csv(r"../TrainingData/ExperimentalTrainingSet.csv",sep=',',header=None)
time = df_full.iloc[0,1:]/3600
idealTime = np.append([0,0], time.iloc[2:,].values)
labels = df_full.iloc[1:,0]
df_data = df_full.iloc[1:,1:]

df_train, df_test = train_test_split(df_data, test_size=test_split, train_size=1-test_split, random_state=42, shuffle=True, stratify=labels)

# normalise time series data
min_value, max_value = df_train.min().min(), df_train.max().max()
df_norm_train = (df_train - min_value)/(max_value - min_value)
df_norm_test = (df_test - min_value)/(max_value - min_value)

if TestOrTrainDataset == 'train':
    X_train = df_norm_train.iloc[:,:SequenceLength].values
    y_train = df_norm_train.iloc[:,SequenceLength-1].values
    X_train = np.expand_dims(X_train, 2)
    y_train = np.broadcast_to(y_train[:,None], (y_train.shape[0],SequenceLength))
    y_train = np.expand_dims(y_train, 2)
    X = X_train
    y = y_train
elif TestOrTrainDataset == 'test':
    X_test = df_norm_test.iloc[:,:SequenceLength].values
    y_test = df_norm_test.iloc[:,SequenceLength-1].values
    X_test = np.expand_dims(X_test, 2)
    y_test = np.broadcast_to(y_test[:,None], (y_test.shape[0],SequenceLength))
    y_test = np.expand_dims(y_test, 2)
    X = X_test
    y = y_test
    
Mean = tf.zeros([len(X),SequenceLength])
Prediction = tf.zeros([len(X),SequenceLength])

for i in range(NumEnsemble):
    checkpoint_filepath = "../Models/EnsembleModel" + str(i+1)
    bestModel = keras.models.load_model(checkpoint_filepath, custom_objects={"MeanVarianceLogLikelyhoodLoss": MeanVarianceLogLikelyhoodLoss})
    loss = bestModel.evaluate(X, y, batch_size=batch_size)    
    # predicts all time series at once, but for some reason has up to a ~5% error compared to calling.predict() on one example
    # perhaps model.predict() is not always deterministic? This is fine for visualisation, but not for real predictions
    # for some reason predicting on X_train[0:6,::] for example does give identical results to predicting any one indivdually..
    Prediction = bestModel.predict(X)
    

    # make a prediction for every time series in X dataset individually, remove extraneous dimensions and convert to np.array
    # Prediction = np.squeeze([bestModel.predict(np.expand_dims(X[i,::], 0)) for  i in range(len(X))])
    
    Mean += Prediction[:,:,0]
        
Mean /= NumEnsemble

PredictionT90 = []
ExperimentalT90 = []

for mean,x in zip(Mean,X):
    
    # Boolean 1D array of when the model prediction or experimental response was within 10% of
    # the final equilibrium experimental response
    PredictionEquilibriumSeries = np.array((mean > x[-1]*(1-t90Window))&(mean < x[-1]*(1+t90Window)))
    ExperimentalEquilibriumSeries = np.array((x > x[-1]*(1-t90Window))&(x < x[-1]*(1+t90Window)))

    # finds the latest time point at which the model was outisde the range of final equilirium response +/- 10%
    PredictionT90.append(SequenceLength - np.argmin(np.flip(PredictionEquilibriumSeries)))
    ExperimentalT90.append(SequenceLength - np.argmin(np.flip(ExperimentalEquilibriumSeries)))
    
PredictionT90 = [t90 * time.iloc[-1]/250 for t90 in PredictionT90]
ExperimentalT90 = [t90 * time.iloc[-1]/250 for t90 in ExperimentalT90]

print('\nprediction t90 summary statistics')
print(pd.DataFrame(PredictionT90).describe())
print('\nexperimental t90 summary statistics')
print(pd.DataFrame(ExperimentalT90).describe())
print('\nratio of experimental to predicted t90 summary statistics')
print(pd.DataFrame(np.array(ExperimentalT90)/np.array(PredictionT90)).describe())

fig, ax = plt.subplots()
ax.hist(PredictionT90, histtype="stepfilled", bins=25, alpha=0.7,label='Prediction t90')
ax.hist(ExperimentalT90, histtype="stepfilled", bins=25, alpha=0.7,label='Experimental t90')

ax.set_ylabel('Frequency', fontsize = 12)
ax.set_xlabel('Time (hours)', fontsize = 12)

ax.tick_params(right=True, top=True, labelright=False, labeltop=False)
ax.tick_params(direction="in")
ax.set_xticks(range(14))
ax.legend(loc='upper right')

plt.savefig(".."+"/Figures/PredictedExperimentalt90Histogram.tif", dpi=200, bbox_inches='tight')
plt.show()