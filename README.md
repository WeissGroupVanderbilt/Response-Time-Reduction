&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/OverviewFigure.png" width = "750" />

# Sensor Response Time Reduction


***
For further details, see the following publications:

Ward, S. J., M. Baljevic, & Weiss, S. M. (2024). “Sensor Response-Time Reduction using Long-Short Term Memory Network Forecasting,”  _Manuscript in preparation_.

Ward, S. J., & Weiss, S. M. (2023). Reduction in sensor response time using long short-term memory network forecasting. Proc. SPIE, 12675(126750E). doi: [10.1117/12.2676836](https://doi.org/10.1117/12.2676836)

***
## Table of Contents
### 1. Motivation
### 2. Dataset
##### 2.1 Experimental Data
##### 2.2 Simluted Data
### 3. 
##### 3.1 Settings
### 5. Troubleshooting
### 6. FAQ
***
## 1. Motivation

The response time of a biosensor is a crucial metric in safety-critical applications such as medical diagnostics where an earlier diagnosis can markedly improve patient outcomes. However, the speed at which a biosensor reaches a final equilibrium state can be limited by poor mass transport and long molecular diffusion times that increase the time it takes target molecules to reach the active sensing region of a biosensor. While optimization of system and sensor design can promote molecules reaching the sensing element faster, a simpler and complementary approach for response time reduction that is widely applicable across all sensor platforms is to use time-series forecasting to predict the ultimate steady-state sensor response. In this work, we show that ensembles of long short-term memory (LSTM) networks can accurately predict equilibrium biosensor response from a small quantity of initial time-dependent biosensor measurements, allowing for significant reduction in response time by a mean and median factor of improvement of 18.6 and 5.1, respectively. The ensemble of models also provides simultaneous estimation of uncertainty, which is vital to provide confidence in the predictions and subsequent safety-related decisions that are made. This approach is demonstrated on real-time experimental data collected by exposing porous silicon biosensors to buffered protein solutions using a multi-channel fluidic cell that enables the automated measurement of 100 porous silicon biosensors in parallel. The dramatic improvement in sensor response time achieved using LSTM network ensembles and associated uncertainty quantification opens the door to trustworthy and faster responding biosensors, enabling more rapid medical diagnostics for improved patient outcomes and healthcare access, as well as quicker identification of toxins in food and the environment.
***
## 2. Dataset

X
### 3.1 Settings
- X
- 
***

## 6. FAQs

***
## 7. Acknowledgements


***
