# ai-project4

## Dev Setup
1. Choose the project you would like to run (`cnn.py` or `cnn_cifar.py`) and execute: <br/>
   `python3 cnn.py` <br/>
   or <br/>
   `python3 cnn_cifar.py`

Note: You can engage a VAE model for the `cnn_cifar.py` file by using `--vae` or `--vae_by_channel`.

### 1D-CNN for IMDB dataset
After implementing the LSTM RNN on the IMDB dataset for sentiment analysis, we only achieved moderately high accuracy. We suspect that the lack of inherent long-range dependencies in sentiment analysis will lend better to the implementation of a 1D covnet CNN. This model will better capture the local dependencies by considering features in smaller chinks, rather than extrapolating dependencies based on non-local tokens.

#### Dropout
We used a dropout rate of 20%. Using dropout increases the robustness of the model, forcing it not to rely on one path for a classification, instead allowing for multiple paths to the correct classification. We believe that a rate of 20% is appropriate based on testing with various values for the dropout hyperparameter.

#### Vocab Size
We chose a vocabulary size of 20,000 in order to fully represent as much of the data set as possible. Since a majority of English sentences use the same limited vocabulary, we can model almost the entirety of the data set with a vocabulary of 20,000. While having a larger vocabulary may better represent the dataset (i.e. less unknown tokens), it also increases computational costs. Therefore, limiting the vocabulary to 20,000 allows relatively quick computations without sacrificing a large amount of accuracy. 

#### Filter
We chose to use 250 filters. We chose this high number of filters because it will allow us to detect more features within the data. This will be important in understanding the context of the input amongst the values to which it is adjacent, but also to the values that are influential to the current token on broader scale. Ultimately, this should be the advantage of using a 1D-CNN architecture over something like an RNN. We can look at more of the search space when trying to understand the relationship amongst input tokens.

#### Kernel Size
We chose to use a kernel size of 3. This will allow for the extraction of finer features necessary in sentiment analysis. Essentially, the model will consider words in groups of three when extracting features. *** Maybe add some more to this

#### Batch Size
We chose a batch size of 64 due to still fast computation speeds, but accurate results. A larger batch size would train faster, but sacrifice accuracy. A smaller batch size will converge more accurately to the local minima, however at higher computational costs. Thus, we chose a batch size that will allow training in a reasonable time, while not sacrificing too much accuracy. 

#### Model Architecture
After embedding out dataset using a input dimension of 20,000 and an output dimension of 64, we implemented our 1D covnet. This layer used the number of filters and kernel size as mentioned previously, as well as a valid padding with a stride of 1 and the relu activation function. Since our dataset is textual, we used valid padding. We used a stride of 1 so that each combination of local tokens is considered. Following the convolution layer, we implemented a max pooling layer. This layer is responsible for down sampling the data and extracting the most prominent features. We chose a max pooling instead of average pooling so that the most prominent features are properly highlighted without being skewed by other local features. Lastly, we implemented a dense layer with one node utilizing the sigmoid activation function. This layer is responsible for using the features extracted by the convolution and pooling layer to generate a prediction. Since we are doing sentiment analysis, the dense layer has one node, where a score closer to one indicates positive sentiment and a score closer to zero indicates negative sentiment. 

#### Results
As we predicted, the 1D covnet had much greater accuracy than the LSTM. Our final model had an accuracy of 89%.


### CNN for CIFAR-10 dataset

#### Filters
After experimenting with various depths, we ended with a depth of size 32. This depth allows for high accuracy at reasonable training times. It is important to note that the depth does not remain consistent throughout the model. We increase depth the further the input moves into the model. This is because the data that is being propagated is becoming increasing complex. The convolutional layers earlier in the model extract these features, and the layers later in the model need to further extrapolate upon them. This means that we need to increase the number of filters to deal with this increasing complexity and to increase the complexity of the features we derive.

#### Epochs
In order to determine the appropriate number of epochs, we ran our model until we saw that the accuracy was leveling out. We found that the accuracy of the model was still underfitting to the data around 50 epochs -- the loss function was still being minimized. So we doubled the number of epochs and found that our accuracy was able to increase to around 84%.

#### Padding
We used same padding. Same padding results in the convolved image to be the same size as the original image. In addition, same padding allows features on the edges of the image to be considered equally to those in the center of the image. Since our dataset consists of images, same padding is advantageous. 

#### Kernel Size
We used a kernel size of 3x3 paired with a stride of 1. Using a 3x3 kernel allows the model to extract prominent local features by considering a small window size. We use a stride of 1 in order to fully consider all local dependencies without skipping over any. 

#### Model Architecture
Out model has three basic parts.

The first takes the images as input and performs two convolusions before pooling. Each convolution uses same padding, a 3x3 kernel size, and a variable number of filters based on depth within the model. We chose to have two convolusion layers before pooling in order to fully identify the features within the image before the loss of information that is inherent in down sampling. We used max pooling to down sample the data with 2x2 dimensions. We used max pooling to highlight the most prominent features without skewing them lower as average pooling would do. In order to create a more robust model, we used a dropout rate of 0.25 and a batch normalization. We found that the combination of the two techniques was more effective than either of the techniques alone. 

The second implements two additional convolusional layers then a max pooling layer. This part is identical to the first, except that the convolusional layers use a batch size of 64. As described previously, this allows the model to further extract the more prominent features that have been identified by the first part of the model. 

The final part of the model implements two deep layers. The first deep layer has 512 nodes and uses the relu activation function. This layer is responsible for using the features identified by the convolusional layers to identify global dependencies within the image. The final layer has 10 nodes, one for each possible image classification. This layer uses a softmax activation function, as used for classification tasks. 

#### Results
Using the model described above, we were able to obtain an accuracy of 84% after 100 epochs. 


#### Using a Variational Autoencoder to augment dataset size
We decided to attempt to use a Variational Autoencoder to increase the size of the Cifar-10 dataset. We believed that this would allow us to introduce slightly varied images to the ones that are currently in the dataset. This would allow the CNN model to be trained not only one images that are in the dataset, but images that have features like -- but not identical too -- those found in the dataset. This would better normalize the model against variation than simply dropout and batch normalization.

We found, however, that introducing a VAE model was not only expensive in terms of time and resources, but was also largely ineffective at generating increased accuracy of the CNN model. This may be due to the VAE model that we chose to use. We used a Dense Autoencoder which flattened each channel of our RGB images into an individual tensor of size 1024 (32x32). So, the entire dataset size tripled from 50,000 images to 150,000 images, because of the division on each channel. We then had to recombine the images into their full form by concatenating the broken down tensors into a tensor of size (32, 32, 3).

This is problematic. Breaking the images apart in this way will tune the weights different based on parts of the same image. It introduces a large amount of variablity to a model which for the most part struggles to deal with varied input. In turn, the loss function did not reduce towards zero nearly as much as we would hope -- even when trained for 200+ epochs. It would be much wiser for us to use a Convolutional Variational Autoencoder, so that we can deal with the RGB channels within the context of a single image.

This network can be run using the `--vae` argument at execution.

#### Using Multiple Variational Autoencoders for channel-level variability
Given the poor results we were able to achieve using the VAE composed of dense layers, we decided to break each image into its component channels and train a VAE for each channel. We thought that this would decrease the variablity that was being introduced to the VAE by the varied channel pixel values. 

Ultimately, this proved to be more expensive and time consuming than it was worth. Not only does splitting the image apart require an O(n^3) operation, but training three independent VAE takes a long time. Furthermore, we were not able to increase accuracy using this technique. The model was able to achieve 80% accuracy over 50 training epochs of the CNN (each VAE was encoded for 50 epochs).

This network can be run using the `--vae_by_channel` argument at execution.

#### Why did VAE not work
We were disappointed by the performance of the VAE model, however, it was not entirely unexpected. Using a Dense Autoencoder to represent images does not exactly make sense as a premise. Images have relationship amongst pixels that are lost on dense layers, due to the nature of data representation within them. They cannot describe edges, corners, etc. with much accuracy at all. This is why our loss for any VAE model we ran was around 650 units. This tells us that the images we were producing did not very closely approximate the input examples at all. This application is much better suited for a Convolutional VAE.

### Sources

https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363

https://medium.com/@ayeshmanthaperera/what-is-padding-in-cnns-71b21fb0dd7

https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/

https://www.quora.com/How-do-I-choose-the-appropriate-batch-size-while-training-a-CNN

https://keras.io/examples/imdb_cnn/

https://keras.io/layers/embeddings/

https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

https://keras.io/utils/

https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e
