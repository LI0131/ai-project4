# ai-project4

## Dev Setup
1. Choose the project you would like to run (`cnn.py` or `cnn_cifar.py`) and execute: <br/>
   `python3 cnn.py` <br/>
   or <br/>
   `python3 cnn_cifar.py`

### 1D-CNN for IMDB dataset

In this part of the project, we use a 1D-CNN to conduct sentiment analysis. We use a Dense layer of size 1 with the sigmoid activation function as our output layer in order to do this. The sigmoid function bounds the incoming signal to be between 0 and 1, so more positive sentiments will drive values which tend closer to 1, while more negative sentiments drive values which are closer to 0.

We were able to achieve an accuracy of approx. 88% using a 1D-CNN model. We downloaded the dataset from the keras API which has a built in testing and training split of 50,000 training examples and 10,000 testing examples. We embed each input example using a keras tensorflow Embedding layer with the input dimension being the size of our vocabulary and the output dimension set to size (20,000, 64) -- that is each word is embedded as a size 64 vector. 


### CNN for CIFAR-10 dataset
