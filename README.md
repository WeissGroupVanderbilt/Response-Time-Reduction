&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/OverviewFigure.png" width = "600" />
# Sensor Response Time Reduction


***
For further details, see the following publications:

Ward, S. J., M. Baljevic, & Weiss, S. M. (2024). “Sensor Response-Time Reduction using Long-Short Term Memory Network Forecasting,”  _Manuscript in preparation_.

Ward, S. J., & Weiss, S. M. (2023). Reduction in sensor response time using long short-term memory network forecasting. Proc. SPIE, 12675(126750E). doi: [10.1117/12.2676836](https://doi.org/10.1117/12.2676836)

***
## Table of Contents
### 1. Motivation
### 2. Experimental Data
### 3. Models
##### 3.1 Settings
### 5. Troubleshooting
### 6. FAQ
***
## 1. Motivation

The response time of a biosensor is a crucial metric in safety-critical applications such as medical diagnostics where an earlier diagnosis can markedly improve patient outcomes. However, the speed at which a biosensor reaches a final equilibrium state can be limited by poor mass transport and long molecular diffusion times that increase the time it takes target molecules to reach the active sensing region of a biosensor.

While optimization of system and sensor design can promote molecules reaching the sensing element faster, a simpler and complementary approach for response time reduction that is widely applicable across all sensor platforms is to use time-series forecasting to predict the ultimate steady-state sensor response.

In this work, we show that ensembles of long short-term memory (LSTM) networks can accurately predict equilibrium biosensor response from a small quantity of initial time-dependent biosensor measurements, allowing for significant reduction in response time by a mean and median factor of improvement of 18.6 and 5.1, respectively. The ensemble of models also provides simultaneous estimation of uncertainty, which is vital to provide confidence in the predictions and subsequent safety-related decisions that are made.

This approach is demonstrated on real-time experimental data collected by exposing porous silicon biosensors to buffered protein solutions using a multi-channel fluidic cell that enables the automated measurement of 100 porous silicon biosensors in parallel. The dramatic improvement in sensor response time achieved using LSTM network ensembles and associated uncertainty quantification opens the door to trustworthy and faster responding biosensors, enabling more rapid medical diagnostics for improved patient outcomes and healthcare access, as well as quicker identification of toxins in food and the environment.
***
## 2. Experimental Data
Porous silicon sensors were fabricated, secured in a multi-channel fluidic cell, and real time optical reflectance measurements were carried out for each sensor in turn as the protein bovine serum albumin (BSA) in buffer solutions (HEPES), at concentrations of 40, 20, 10, 4, 2, 1, 0.4, 0.2, 0.1, 0.04, 0.02, 0.002 mg/ml, and a control solution consisting of 100% buffer, were dropped onto the sensors.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/BSA.png" width = "400" /> 

Collection of a sufficiently large dataset was enabled by using an automated real-time measurement setup in which the multi-channel fluidic cell was affixed to a stepper motor, which was controlled using python to cycle through the reflectance measurement of many sensors in sequence.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/HighThroughputMeasurementSetup.png" width = "700" />
***
## 3. Models
LSTM networks are well suited for the rapid prediction of equilibrium sensor response due to their ability to learn features without requiring manual feature selection, to learn to distinguish signal from noise, and to learn long and short term dependencies in sequential data, all of which promote generalizability. 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/LSTM.png" width = "700" />

Models were implemented in tensorflow using the keras API and built-in LSTM layers. Each LSTM layer was configured to return a sequence of 250 outputs, one for each input time step. The target output, used to compute the loss, is the final element of each input sequence, repeated in a vector with 250 elements. Each output prediction in the sequence is made having only seen data from the current and past timesteps, so will typically become increasingly more accurate as the sequence goes on and more data from the input sequence is seen by the model.

The data was first randomly shuffled and split into train, validation and test sets at a ratio of 3:1:1, stratified by BSA concentration. Ensembles of 15 base learners were trained in turn, by minimizing the negative log likelihood (
−log p(y|x) ), using softplus activation at the output layer to ensure predictions are positive, and adam optimization. Ensembles were used to increase accuracy and prediction stability, and for better calibrated uncertainty quantification.

The base learner architecture, informed by limited hyperparameter tuning using the validation set, was the following: 50 input neurons, 1 hidden layer with 500 neurons, and 2 output neurons. The maximum and minimum sensor response values across all time steps and all examples in the training set were used to normalize the train, validation and test sets, to avoid data leakage.
***
### 3.1 Settings
- X
- 
***

## 6. FAQs

***
## 7. Acknowledgements


***
