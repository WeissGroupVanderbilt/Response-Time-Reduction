import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from LSTMutils import MeanVarianceLogLikelyhoodLoss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
NumEpochs = 2000
StdevsFromMean = 2 #1, 2 and 3 standard deviations away from the mean encompasses 68%, 95% and 99.7% of the distribution, respectively 
TestOrTrainDataset = 'test' #'train' or 'test'
t90Window = 0.1 # t90 is defined as the time until the response settles to within this fraction of the equilibrium response indefinitely
test_split = 0.2
markerSize = 50

# read experimental data and split into time (s) converted to hours, concentration labels and time series data
df_full = pd.read_csv(r"../TrainingData/ExperimentalTrainingSet.csv",sep=',',header=None,index_col=0)
time = df_full.iloc[0,:]/3600
idealTime = np.append([0,0], time.iloc[2:,].values)
labels = df_full.index[1:]
df_data = df_full.iloc[1:,:]


df_train, df_test = train_test_split(df_data, test_size=test_split, train_size=1-test_split, random_state=42, shuffle=True, stratify=labels)

# normalise time series data
min_value, max_value = df_train.min().min(), df_train.max().max()
df_norm_train = (df_train - min_value)/(max_value - min_value)
df_norm_test = (df_test - min_value)/(max_value - min_value)



concentrations = ["2","0.002","0.02","0.2","0.4","1","4","10","20","0.1","0.04","40"]
fig, ax = plt.subplots()
MedianPredictionT90LSTM = []
MedianPredictionT90RNN = []
MedianPredictionT90GRU = []
MedianExperimentalT90 = []
RatioT90LSTM = []
RatioT90RNN = []
RatioT90GRU = []

for concentration in concentrations:

    
    if TestOrTrainDataset == 'train':
        X_train = df_norm_train.loc[concentration,:SequenceLength].values
        y_train = df_norm_train.loc[concentration,SequenceLength-1].values
        X_train = np.expand_dims(X_train, 2)
        y_train = np.broadcast_to(y_train[:,None], (y_train.shape[0],SequenceLength))
        y_train = np.expand_dims(y_train, 2)
        X = X_train
        y = y_train
    
    elif TestOrTrainDataset == 'test':
        X_test = df_norm_test.loc[concentration,:SequenceLength].values
        y_test = df_norm_test.loc[concentration,SequenceLength-1].values
        X_test = np.expand_dims(X_test, 2)
        y_test = np.broadcast_to(y_test[:,None], (y_test.shape[0],SequenceLength))
        y_test = np.expand_dims(y_test, 2)
        X = X_test
        y = y_test
    
    NumGoodModels = 0
    MeanLSTM = tf.zeros([len(X),SequenceLength])
    MeanRNN = tf.zeros([len(X),SequenceLength])
    MeanGRU = tf.zeros([len(X),SequenceLength])
    PredictionLSTM = tf.zeros([len(X),SequenceLength])
    PredictionRNN = tf.zeros([len(X),SequenceLength])
    PredictionGRU = tf.zeros([len(X),SequenceLength])
    
    for i in range(NumEnsemble):
        checkpoint_filepathLSTM = "../Models/EnsembleModel" + str(i+1)
        bestModelLSTM = keras.models.load_model(checkpoint_filepathLSTM, custom_objects={"MeanVarianceLogLikelyhoodLoss": MeanVarianceLogLikelyhoodLoss})
        
        checkpoint_filepathRNN = "../Models/AlternativeModels/RNN/EnsembleModel" + str(i+1)
        bestModelRNN = keras.models.load_model(checkpoint_filepathRNN, custom_objects={"MeanVarianceLogLikelyhoodLoss": MeanVarianceLogLikelyhoodLoss})

        checkpoint_filepathGRU = "../Models/AlternativeModels/GRU/EnsembleModel" + str(i+1)
        bestModelGRU = keras.models.load_model(checkpoint_filepathGRU, custom_objects={"MeanVarianceLogLikelyhoodLoss": MeanVarianceLogLikelyhoodLoss})
        
        
        lossLSTM = bestModelLSTM.evaluate(X, y, batch_size=batch_size)
        lossRNN = bestModelRNN.evaluate(X, y, batch_size=batch_size)
        lossGRU = bestModelGRU.evaluate(X, y, batch_size=batch_size)
        # predicts all time series at once, but for some reason has up to a ~5% error compared to calling.predict() on one example
        # perhaps model.predict() is not always deterministic? This is fine for visualisation, but not for real predictions
        # for some reason predicting on X_train[0:6,::] for example does give identical results to predicting any one indivdually..
        PredictionLSTM = bestModelLSTM.predict(X)
        PredictionRNN = bestModelRNN.predict(X)
        PredictionGRU = bestModelGRU.predict(X)
        
        # make a prediction for every time series in X dataset individually, remove extraneous dimensions and convert to np.array
        # Prediction = np.squeeze([bestModel.predict(np.expand_dims(X[i,::], 0)) for  i in range(len(X))])

        MeanLSTM += PredictionLSTM[:,:,0]
        MeanRNN += PredictionRNN[:,:,0]
        MeanGRU += PredictionGRU[:,:,0]
        NumGoodModels += 1

    MeanLSTM /= NumGoodModels
    MeanRNN /= NumGoodModels
    MeanGRU /= NumGoodModels
    PredictionT90LSTM = []
    PredictionT90RNN = []
    PredictionT90GRU = []
    ExperimentalT90 = []
    
    for meanlstm,x,meanrnn,meangru in zip(MeanLSTM,X,MeanRNN,MeanGRU):

        # Boolean 1D array of when the model prediction or experimental response was within 10% of
        # the final equilibrium experimental response
        PredictionEquilibriumSeriesLSTM = np.array((meanlstm > x[-1]*(1-t90Window))&(meanlstm < x[-1]*(1+t90Window)))
        PredictionEquilibriumSeriesRNN = np.array((meanrnn > x[-1]*(1-t90Window))&(meanrnn < x[-1]*(1+t90Window)))
        PredictionEquilibriumSeriesGRU = np.array((meangru > x[-1]*(1-t90Window))&(meangru < x[-1]*(1+t90Window)))
        ExperimentalEquilibriumSeries = np.array((x > x[-1]*(1-t90Window))&(x < x[-1]*(1+t90Window)))
        
        # finds the latest time point at which the model was outisde the range of final equilirium response +/- 10%
        PredictionT90LSTM.append(SequenceLength - np.argmin(np.flip(PredictionEquilibriumSeriesLSTM)))
        PredictionT90RNN.append(SequenceLength - np.argmin(np.flip(PredictionEquilibriumSeriesRNN)))
        PredictionT90GRU.append(SequenceLength - np.argmin(np.flip(PredictionEquilibriumSeriesGRU)))
        ExperimentalT90.append(SequenceLength - np.argmin(np.flip(ExperimentalEquilibriumSeries)))

        time = np.array(time, dtype=float)
    
    medianpredictiont90lstm = np.median(PredictionT90LSTM)
    medianpredictiont90rnn = np.median(PredictionT90RNN)
    medianpredictiont90gru = np.median(PredictionT90GRU)
    medianexperimentalt90 = np.median(ExperimentalT90)
    ratiot90lstm = medianexperimentalt90/medianpredictiont90lstm
    ratiot90rnn = medianexperimentalt90/medianpredictiont90rnn
    ratiot90gru = medianexperimentalt90/medianpredictiont90gru
    
    MedianPredictionT90LSTM.append(medianpredictiont90lstm)
    MedianPredictionT90RNN.append(medianpredictiont90rnn)
    MedianPredictionT90GRU.append(medianpredictiont90gru)
    MedianExperimentalT90.append(medianexperimentalt90)
    
    RatioT90LSTM.append(ratiot90lstm)
    RatioT90RNN.append(ratiot90rnn)
    RatioT90GRU.append(ratiot90gru)
    
    Concentration = float(concentration)
    ax.scatter(Concentration,ratiot90lstm, facecolor=(0,0,0,0), edgecolor="tab:blue", marker='s', s=markerSize-20)
    ax.scatter(Concentration,ratiot90rnn, facecolor=(0,0,0,0), edgecolor="tab:orange", marker='^', s=markerSize)
#     ax.scatter(Concentration,ratiot90gru, facecolor=(0,0,0,0), edgecolor="g", marker='o', s=markerSize)
 
Concentrations = [float(i) for i in concentrations]

LinearFit = LinearRegression()
LinearFit.fit(np.reshape(Concentrations,(-1, 1)), np.reshape(RatioT90LSTM,(-1, 1)))
ax.plot(Concentrations,LinearFit.predict(np.reshape(Concentrations,(-1, 1))),label="LSTM",color="tab:blue")
LinearFit.fit(np.reshape(Concentrations,(-1, 1)), np.reshape(RatioT90RNN,(-1, 1)))
ax.plot(Concentrations,LinearFit.predict(np.reshape(Concentrations,(-1, 1))),label="RNN",color="tab:orange")
# LinearFit.fit(np.reshape(Concentrations,(-1, 1)), np.reshape(RatioT90GRU,(-1, 1)))
# ax.plot(Concentrations,LinearFit.predict(np.reshape(Concentrations,(-1, 1))),label="GRU",color="tab:grreen")

ax.set_ylabel('Ratio of Experimental to \nPredicted Response Time', fontsize = 12)
ax.set_xlabel('Concentration (g/L)', fontsize = 12)

plt.legend()
plt.grid(which='major',linestyle='--')

plt.savefig("../Figures/t90ReductionLSTMvsRNNvsConcentrationMedian.tif", dpi=200, bbox_inches='tight')
plt.show()