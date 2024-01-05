import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

def MeanVarianceLogLikelyhoodLoss(y_true, y_pred):
    Variance = tf.expand_dims(y_pred[:,:,1], 2)
    Variance = tf.math.add(Variance, 1e-6)
    Mean = tf.expand_dims(y_pred[:,:,0], 2)
    #tf.math.log is base e by default
    return tf.math.log(Variance)/2 + (y_true-Mean)**2/(2*Variance)

def RedlichPetersonIsotherm(concentration, A, B, beta, I):
    return I + A*concentration/(1 + B*(concentration**beta))

class ExperimentalData:
    "A class to read and preprocess experimental data"
    def __init__(self,SequenceLength=None,ExperimentalDataFilePath = "../TrainingData/ExperimentalTrainingSet.csv"
                 , FEMfitParamsFilePath = "../TrainingData/FEMFitParameters.csv"):
        
        self.ExperimentalDataFilePath = ExperimentalDataFilePath
        self.FEMfitParamsFilePath = FEMfitParamsFilePath
        self.SequenceLength = SequenceLength
        
    def ReadData(self,ConcentrationAsIndex=None):
        df_full = pd.read_csv(self.ExperimentalDataFilePath,sep=',',header=None,index_col=ConcentrationAsIndex)
        
        #split data into time converted from seconds to hours, concentration labels and time series data (with/without concentration labels)
        time = df_full.iloc[0,1:]/3600
        concentrations = df_full.iloc[1:,0]
        df_data = df_full.iloc[1:,1:]
        df_data_with_concentrations = df_full.iloc[1:,0:]
        return time, concentrations, df_data, df_data_with_concentrations
    
    def ReadFEMfitParams(self):
        df_full = pd.read_csv(self.FEMfitParamsFilePath,sep=',',header=0)
        
        #remove first column of labels
        df_full = df_full.iloc[:,1:]
        
        concentrations = df_full.iloc[:,0]
        df_data = df_full.iloc[:,1:]
        return concentrations, df_data
    
    def NormalizeData(self,df_train,df_test,df_val=[]):
        min_value, max_value = df_train.min().min(), df_train.max().max()
        df_norm_train = (df_train - min_value)/(max_value - min_value)
        df_norm_test = (df_test - min_value)/(max_value - min_value)
        df_norm_val = (df_val - min_value)/(max_value - min_value)
        return df_norm_train, df_norm_test, df_norm_val
            
    def Shape(self,df_norm):
        X = df_norm.iloc[:,:self.SequenceLength].values
        y = df_norm.iloc[:,self.SequenceLength-1].values
        X = np.expand_dims(X, 2)
        y = np.broadcast_to(y[:,None], (y.shape[0],self.SequenceLength))
        y = np.expand_dims(y, 2)
        return X, y
        
class ModelTrainingEvaluation:
    "A class to evaluate model training"
    def PlotLossHistory(self,history):
        loss_values = history.history['loss']
        val_loss_values = history.history['val_loss']
        epochs = range(1, len(loss_values)+1)
        plt.plot(epochs, loss_values, label='Training Loss')
        plt.plot(epochs, val_loss_values, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()