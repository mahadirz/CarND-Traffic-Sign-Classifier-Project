# **Traffic Sign Recognition** 

## Writeup

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[label_images]: ./writeup_images/label_images.png "label_images"
[histogram]: ./writeup_images/histogram.png "histogram"
[adaptive_equalization]: ./writeup_images/adaptive_equalization.png "adaptive_equalization"
[model_loss]: ./writeup_images/model_loss.png "model_loss"
[model_acc]: ./writeup_images/model_acc.png "model_acc"
[web_images]: ./writeup_images/web_images.png "web_images"
[1]: ./writeup_images/1.png "1"
[2]: ./writeup_images/2.png "2"
[3]: ./writeup_images/3.png "3"
[4]: ./writeup_images/4.png "4"
[5]: ./writeup_images/5.png "5"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set

After the dataset is downloaded from AWS s3 bucket and extracted out, 3 files are found namely `train.p`, `valid.p` and `test.p`.
These files are in pickle format, after the content of files are read into memory,
I have performed preliminary check of its summary specifically 
its sizes, these information are computed using the numpy shape method. Followings are the summaries:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Overall, the percentage split of training,validation and test set is 67%/9%/24% accordingly. 

#### 2. Include an exploratory visualization of the dataset.

To start my exploratory analysis I have plotted random image each for every unique label from 0 till 42. 

![alt text][label_images]

The name of each label of 0 to 42 can be found in `signnames.csv`, for example label 0 is "Speed limit (20km/h)"
and label 17 is a "No Entry" sign. I have used pandas library to merge the `y_train` and `sign_names` and subsequently
plot a histogram to visualize the distribution of the labels. The `seaborn` library is used here instead of matplotlib as it
offers more simplify API for plotting.

![alt text][histogram]

From the histogram above, we can see the distribution of the labels are not equal. For training deep learning model having many examples and equal size of
each class is a common practice, from simple internet search, I have found several methods to do this but I didn't pursue it due to time contraint
and my mo... TODO

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data

I have iterated several methods to use in processing such as the simple normalization and grayscaling the image, however
I found the best method is normalization through the histogram adaptive equalization. From my own testing this technique has improved
the accuracy of predicting image taken from Google search as well as reduce the time for model to converge.
The technique is found after reading article from [Dr. Adrian regarding traffic sign classifier](https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/)

The following picture compares the image with and without the technique being applied. The top images are the original.

![alt text][adaptive_equalization]

#### 2. Describe what your final model architecture looks like

For the model architecture I have adopted the LetNet5 with minor modifications. The original model does seem to perform
quite well but in order to achieve minimal 93% accuracy requirement I have modified the input to be 3 channels of RGB 
colors instead of only single. After several iteration of experiments by changing the architecture I found by adding an additional
convolutional layer is pretty much enough to achieve the desired accuracy. To reduce over fitting the dropout is also introduced
after the first fully connected layer.

My final model consisted of the following layers:

| Layer                 | Description                                   |
|-----------------------|-----------------------------------------------|
| Input                 | 32x32x3 RGB Image                             |
| Convolutional 5x5     | 1x1 stride, valid padding. Output 28x28x6.    |
| RELU                  | Activation function                           |
| Convolutional 2x2     | 2x2 stride, valid padding. Output 14x14x6     |
| RELU                  | Activation function                           |
| Convolutional 5x5     | 1x1 stride, valid padding. Output 10x10x16    |
| RELU                  | Activation function                           |
| Max pooling 2x2       | 2x2 stride, valid padding.Output 5x5x16       |
| Flatten               | Output 1x400                                  |
| Fully Connected Layer | Output 1x120                                  |
| RELU                  | Activation function                           |
| Dropout               | Regularization layer                          |
| Fully Connected Layer | Output 1x84                                   |
| RELU                  | Activation function                           |
| Fully Connected Layer | Output 1x43                                   |
| Softmax               | Produce one hot encoded softmax cross entropy |


#### 3. Describe how you trained your model

The hyperparameters used for training are as follows:

Epoch: 50
Batch size: 128
Learning rate: Decay with starting value 0.001
Dropout: 50%

The learning rate decay is adopted from this [article](https://ireneli.eu/2016/03/13/tensorflow-04-implement-a-lenet-5-like-nn-to-classify-notmnist-images/).
The intuition behind this approach is at the beginning of training the rate can be set a little bit higher to allow the descend to progress faster
but after several iterations the value can be reduced to avoid converging to local minimum.

I have also used the default `AdamOptimizer` from the the lesson with default values except for the learning rate.

#### 4. Describe the approach taken for finding a solution

For the final solution, The training accuracy and loss comparison between training and validation are shown below:

![alt text][model_loss]
![alt text][model_acc]

The training took 30 minutes using my Macbook Pro for 50 epochs. Looking at chart the validation accuracy achieved at least
93% at the 7th epoch.

After the training completed the model's weights are saved into a `model` folder. I have create a utility function
to load these weights into Tensorflow session to perform prediction.

```python
def predict(X, n=5, mode_dir='model'):
    """
    Predict the data
    return the predicted class and probability
    i.e: top_k.indices[0][0], top_k.values[0][0]
    is the predicted class with its probability
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(mode_dir))
        #softmax = sess.run(tf.nn.softmax(logits), feed_dict={x: X, keep_prob: 1.0})
        top_k = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=n), feed_dict={x: X, keep_prob: 1.0})
        return top_k
```

This function is then used to compute the accuracy of test set, following are the summary of accuracies for final result:

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 96.9%
* test set accuracy of 94.4%

To achieve above accuracy I used an iterative approach to experiment with the LetNet5 architecture.

#### 1st iteration

* Use the same LetNet architecture from the lesson
* The traffic sign images are converted from RGB to Grayscale
* The output layer is change from 10 to 43
* Use 10 epochs, batch size 128 and learning rate of 0.001
* The training accuracy 98.2% and validation accuracy 87.4%

#### 2nd Iteration

* Image channels are retained, training with RGB image
* The hypothesis here the model might use the color to construct more features to recognize the traffic sign.
* The first input_depth convolutional layer is changed from 1 to 3.
* Use 10 epochs, batch size 128 and learning rate of 0.001
* The training accuracy 98.2% and validation accuracy 85.1%
* The validation is reduced from the first iteration, the model might be underfitted.
* By adding more data, i.e from 32x32x1 to 32x32x3 input, the model may require more layers.

#### 3rd Iteration

* The hypothesis at this stage is by adding more convolutional layers, I might solve the underfit.
* Without any complex modification I just start by replacing the first max pooling layer to convolutional layer 
* The total convolutional layers increased from 2 to 3.
* The validation accuracy breakthrough the 93% after 25th epoch
* The Loss chart however shows the training vs validation is diverging. This may be due to overfit.
* The validation loss also have high noise of up and down. This may suggest learning rate is too high
* ![alt text][diverge]

#### 4th Iteration

* Introduce dropout after the first fully connected layer to reduce the overfitting.
* Use Learning rate decay to adjust the learning rate dynamically
* Experiment with normalization to speed up the converging.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web.

The 5 images are taken from [alamy](https://www.alamy.com ) and [rgbstock](https://www.rgbstock.coma) and manually
cropped into the region that contains the traffic sign. In Addition, extra preprocessing is done into the image
to resize it into 32x32. 
Here are five German traffic signs that I found on the web after resized:

![alt text][web_images]

The quality of all the images before the resize are pretty good, with human eyes all images can be classified correctly
without problem. 

As for the model, I suspect it may have hard time to predict due to the rotation/tilt and background of the images might
have been different from what is seen from the training.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set

Here are the results of the prediction:

| Image                | Prediction    |
|----------------------|---------------|
| No passing           | No passing    |
| No entry             | No entry      |
| Turn left ahead      | Keep right    |
| Road work            | Slippery road |
| Speed limit (30km/h) | No passing    |

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 
This does not compares favorably to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model can be reused from the utility `predict` function as explained in previous section.

```python
# Get the top 5 probabilities
test_pred = predict(X,n=5)
```

For the first image, the model is relatively sure that this is a stop sign (probability of 0.52), 
and the image does contain a No Passing sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .52         			| No passing   									| 
| .43     				| Vehicles over 3.5 metric tons prohibited  	|
| .04					| End of no passing								|
| .008	      			| End of no passing by vehicles over 3.5 metric |
| .004				    | No passing for vehicles over 3.5 metric tons	|

![alt text][1]

For the second image, the model is 100% sure that this is a No entry (probability of 1.0), 
and the image does contain a No Entry sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| No entry   									| 
| .00     				| No passing                                	|
| .00					| Speed limit (20km/h)						    |
| .00	      			| Stop                                          |
| .00				    | Speed limit (30km/h)	                        |

![alt text][2]


For the third image, the model is certain that this is a Keep right (probability of 0.99), 
but the image actually contains a Turn left ahead sign. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Keep right   									| 
| .003     				| Turn left ahead                               |
| .00					| Ahead only				                    |
| .00	      			| Go straight or left                           |
| .00				    | Go straight or right                          |

![alt text][3]


For the fourth image, the model can be said to be uncertain by predicting the Slippery road (probability of 0.45)
The actual image is a Road work. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.45        			| Slippery road  								| 
| .39     				| Dangerous curve to the right                  |
| .16					| Road work 				                    |
| .00	      			| No passing for vehicles over 3.5 metric tons  |
| .00				    | Dangerous curve to the left	                |

![alt text][4]

For the fifth image, the model can be said to be uncertain by predicting the Slippery road (probability of 0.45)
The actual image is a Road work. The top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.35        			| No passing     								| 
| .12     				| Dangerous curve to the left                   |
| .11					| Vehicles over 3.5 metric tons prohibited      |
| .08	      			| Speed limit (30km/h)                          |
| .06				    | Speed limit (20km/h)       	                |

![alt text][5]

### Conclusions

The model can be improved further by revisiting particularly model architecture and image preprocessing. On
the preprocessing it's possible to increase the class that has little example by augmentation such as described in 
[Keras imagedatagenerator](https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/).

Besides that, I don't spend much time on iterating on the model architecture. The simple tweak of replacing the max pooling
layer with convolutional is able to achieve the minimum accuracy but it's not able to generalize well on internet images
this might be an indicator the model might under perform in real world where noises such as 
under various lightning conditions and different angles are unavoidable.



