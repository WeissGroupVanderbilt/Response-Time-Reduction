import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math

def MeanVarianceLogLikelyhoodLoss(y_true, y_pred):
    print
    Variance = tf.expand_dims(y_pred[:,:,1], 2)
    #Variance = Variance + tf.constant(1e-6, shape=Variance.shape)
    Variance = tf.math.add(Variance, 1e-6)
    Mean = tf.expand_dims(y_pred[:,:,0], 2)
    #tf.math.log is base e by default
    return tf.math.log(Variance)/2 + (y_true-Mean)**2/(2*Variance)


class LSTMnetwork:
    "A LSTM network class"
    def __init__(self,X_train,y_train,validation_split,batch_size,epochs,SequenceLength,verbose):
        self.X_train = X_train
        self.y_train = y_train
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.SequenceLength = SequenceLength
        #'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        self.verbose = verbose
        
    def Build(self):
        self.model = keras.models.Sequential([keras.layers.LSTM(300, input_shape=(self.SequenceLength,1),
                                                                return_sequences=True, activation='sigmoid')
                                 , keras.layers.LSTM(100, return_sequences=True, activation='sigmoid')
                                 , keras.layers.LSTM(2, activation='softplus',return_sequences=True)])
    
    def Compile(self):
        self.model.compile(optimizer="adam",loss = MeanVarianceLogLikelyhoodLoss)
        
    def Train(self):
        self.history = self.model.fit(self.X_train, self.y_train, validation_split=self.validation_split
                                      , batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)

    def Evaluate(self,X_test,y_test):
        return self.model.evaluate(X_test, y_test, batch_size=self.batch_size) #min loss shold be -5.988
    
    def Predict(self,sequence):
        self.prediction = self.model.predict(sequence)
        self.X_predict = sequence
    
    def PlotPrediction(self):
        plt.plot(self.prediction.reshape(self.prediction.shape[1],2)[:,0],label="Mean")
        plt.plot(self.prediction.reshape(self.prediction.shape[1],2)[:,1], label="Variance")
        plt.plot(self.X_predict.reshape(self.X_predict.shape[1],1), label="Ground Truth")
        plt.legend()
    
    def PlotTraining(self):
        loss_values = self.history.history['loss']
        epochs = range(1, len(loss_values)+1)
        plt.plot(epochs, loss_values, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()