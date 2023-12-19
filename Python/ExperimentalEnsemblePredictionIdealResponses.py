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

# 
NumEnsemble = 10
SequenceLength = 250
validation_split = 0.25
batch_size = 32
experiment_num = 70
StdevsFromMean = 2 #1, 2 and 3 standard deviations away from the mean encompasses 68%, 95% and 99.7% of the distribution, respectively 
test_split = 0.2

# read experimental data and split into time (s) converted to hours, concentration labels and time series data
df_full = pd.read_csv(r"../TrainingData/ExperimentalTrainingSet.csv",sep=',',header=None)
time = df_full.iloc[0,1:]/3600
labels = df_full.iloc[1:,0]
df_data = df_full.iloc[1:,1:]

# stratified split into train and test sets
df_train, df_test = train_test_split(df_data, test_size=test_split, train_size=1-test_split, random_state=42, shuffle=True, stratify=labels)

# normalise time series data
min_value, max_value = df_train.min().min(), df_train.max().max()
df_norm_train = (df_train - min_value)/(max_value - min_value)
df_norm_test = (df_test - min_value)/(max_value - min_value)

# slicing 
X_test = df_norm_test.iloc[:,:SequenceLength].values
y_test = df_norm_test.iloc[:,SequenceLength-1].values
X_test = np.expand_dims(X_test, 2)
y_test = np.broadcast_to(y_test[:,None], (y_test.shape[0],SequenceLength))
y_test = np.expand_dims(y_test, 2)


Mean = tf.zeros([1,SequenceLength])
Variance = tf.zeros([1,SequenceLength])
X_predict = np.expand_dims(X_test[experiment_num,::], 0)

idealTime = np.append([0,0], time.iloc[2:,].values)
idealPrediction = np.ones(len(idealTime))*X_predict[0,-1,0]
idealPrediction[0] = 0.5

NumGoodModels = 0

for i in range(NumEnsemble):
    checkpoint_filepath = "../Models/EnsembleModel" + str(i+1)
    bestModel = keras.models.load_model(checkpoint_filepath, custom_objects={"MeanVarianceLogLikelyhoodLoss": MeanVarianceLogLikelyhoodLoss})
    loss = bestModel.evaluate(X_test, y_test, batch_size=batch_size,verbose=0)
    if(loss<-2):
        prediction = bestModel.predict(X_predict)
        print(prediction[:,-1,0])
        Mean += prediction[:,:,0]
        Variance += (prediction[:,:,0])**2 + prediction[:,:,1]**2
        NumGoodModels += 1
Mean /= NumGoodModels
Variance /= NumGoodModels
Variance -= Mean**2

Mean = np.squeeze(Mean)
Stdev = np.squeeze(Variance**0.5)
time = np.array(time, dtype=float)

fig, ax = plt.subplots()

ax.fill_between(time, (Mean+Stdev*StdevsFromMean), (Mean-Stdev*StdevsFromMean), alpha=.5, linewidth=0)
ax.plot(time, Mean, linewidth=2, label='Ensemble Mean Prediction')
plt.plot(time,tf.squeeze(X_predict), label="Ground Truth Sensor Response")
plt.plot(idealTime,idealPrediction, label="Ideal Prediction",color='k',linestyle='--')

ax.set_ylabel('Normalized Optical Response', fontsize = 12)
ax.set_xlabel('Time (hours)', fontsize = 12)

ax.tick_params(right=True, top=True, labelright=False, labeltop=False)
ax.tick_params(direction="in")
ax.set_xticks(range(14))
ax.legend()#loc='upper right')

plt.savefig("../Figures/ExperimentalEnsemblePredictionIdealResponses.tif", dpi=200, bbox_inches='tight')
plt.show()