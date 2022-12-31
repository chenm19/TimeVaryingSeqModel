# Time-Varying Sequence Model
This is the code for "Time-Varying Sequence Model" .

## Abstract

Traditional machine learning sequence models, such as RNN and LSTM, can solve se- 1
quential data problems with the use of internal memory states. However, the neuron units and 2
weights are shared at each time step to reduce computational costs, limiting their ability to learn 3
time-varying relationships between model inputs and outputs. In this context, this paper proposes 4
two methods to characterize the dynamic relationships in real-world sequential data, namely the 5
internal time-varying sequence model (ITV model) and the external time-varying sequence model 6
(ETV model). Our methods were designed with an automated basis expansion module to adapt 7
internal or external parameters at each time step without exerting high computational complexity. 8
Extensive experiments performed on synthetic and real-world data demonstrated superior prediction 9
and classification results to conventional sequence models. Our proposed ETV model is particularly 10
effective at handling long sequence data. 

# Requirements

tensorflow (2.4.0)
tensorflow-gpu (2.4.0)
keras (2.6.0)
numpy (1.19.2)
matplotlib (2.2.2)
scikit-fda (0.5)
