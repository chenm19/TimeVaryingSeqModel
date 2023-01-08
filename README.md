# Time-Varying Sequence Model
This is the code for "Time-Varying Sequence Model" .
Contact email: zhaojianxiang777@gmail.com

## Abstract

Traditional machine learning sequence models, such as RNN and LSTM, can solve sequential data problems with the use of internal memory states. However, the neuron units and weights are shared at each time step to reduce computational costs, limiting their ability to learn time-varying relationships between model inputs and outputs. In this context, this paper proposes two methods to characterize the dynamic relationships in real-world sequential data, namely the internal time-varying sequence model (ITV model) and the external time-varying sequence model (ETV model). Our methods were designed with an automated basis expansion module to adapt internal or external parameters at each time step without exerting high computational complexity. Extensive experiments performed on synthetic and real-world data demonstrated superior prediction and classification results to conventional sequence models. Our proposed ETV model is particularly effective at handling long sequence data. 

# Requirements

tensorflow (2.4.0)

tensorflow-gpu (2.4.0)

keras (2.6.0)

numpy (1.19.2)

matplotlib (2.2.2)

scikit-fda (0.5)


# Demos

We also provide two demo notebooks that show how to reproduce some of the results and figures from the paper.

##Simulation data prediction

![image](https://github.com/chenm19/TimeVaryingSeqModel/blob/main/figs/Simulation.png)

The https://github.com/chenm19/TimeVaryingSeqModel/blob/main/Simulation%20data%20prediction.ipynb notebook contains a demonstration and tutorial for experiments comparing the predicted effects of the ITV,ETV model and the original RNN and LSTM on simulation data.




##Real stock data forecast

![image](https://github.com/chenm19/TimeVaryingSeqModel/blob/main/figs/stock.png)

The notebook https://github.com/chenm19/TimeVaryingSeqModel/blob/main/Twitter%20stock%20forecast.ipynb contains a demonstration and tutorial for an experiment comparing the ETV model with the original RNN and LSTM predictions on Twitter stock data.

