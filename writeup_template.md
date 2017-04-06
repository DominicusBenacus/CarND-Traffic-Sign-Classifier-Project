#**Traffic Sign Recognition** 

# Below is my summary of the last 4 weeks. So I spend a lot of time for this project. More than I expected at start. I learned much thing not only CNNs :-). Not everytime just in the content of CNN much time I spend also to solve IT problems according to aws, data transfer and of course my company notebook worked absolutely horrible. In addition the result are not perfect but in combination with this issues I am very happy to solve the most of the requirements. At the end I am a little sad about that there was to less time to take a closer look to the theoretical content of CNN. Additionally I need a good numpy tutorial ;-).

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is?
Number of training examples = 34799 used n_train = y_train.shape[0]

* The size of test set is?
Number of testing examples = 12630 used n_test = y_test.shape[0]

* The shape of a traffic sign image is ?
Image data shape = (32, 32, 3) used image_shape = X_train[0].shape

* The number of unique classes/labels in the data set is?
Number of classes = 43 used sign_classes, class_indices, class_counts = np.unique(y_train, return_index = True, return_counts = True)
n_classes = class_counts.shape[0]

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the In[15] and In[14] code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a histogram over all used classes in the training set. First picture is a ramdom single plot of one traffic sign. 
![][image1]
![][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Code cell of ipynb: [9]:
* At first:
    I decided to convert the images to grayscale because that is a recommondation out of the official LeNet paper. Colored images did not help for better acurracy as Pierre Sermanet and Yann LeCun mentioned in [their papers](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf "Traffic Sign Recognition with Multi-Scale Convolutional Networks")

    There are very much methods to this. I tried out cv2 methods like cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY) and at the end I decided to use a simple factor based method and multiplied each color channel inthe 4d numpz array. To figure out I did implement the know ledge out of this page [this paper] (http://entropymine.com/imageworsener/grayscale/)
* Second step is to normalize --> [9] 
    Why normalization? In common it is easier to train the network and prevent gradient explotion or SGD fails according to converge.
    There are more reasons: It make training faster and reduce the risk of getting stuck in local optima, what is a big risk. But there are many papers about this question. So it depends also on the architecture and the origin data structure. [Here are some more iscussions](http://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network). Overall there is a consense to use normalized data according to LeNet architecture and often according stochastic gradient descent method.

    I did a simple normalization by division like X = (X / 255.-0.5).astype(np.float32) where X is x_train or y_train or... every other function input
* Third step equalization Hist. Like we can see in the picture below. The images are often looks very dark and it is not           possible to detect good edges because of the bad contrast.
    To get better contrast we need a more balanced grayscaled image. So I tried different libs like open cv. Some artifacts are still implemented but out commented in the ipynb. At the end I used the skimage libarz with exposure.equalize_hist(). In origin I wanted to use equalizationAdaphist(), but I found no way to get running it in code. So the equalize_hist is a good solution. Not perfect but it capsulates many issues of the trainng data. 

    One can see the differences between color to grazscale to equaliyation int the below. 
 

Here is an example of a traffic sign image before and aftter the preprocessing steps

![alt text][image3]

My expectation was to get a good accury after these two steps of pre processing. Round about 0.94. But it did not happen as I expacted it.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

So I decided to extend my pre processing method. But because of huge efforts with get running my IT and much problems with aws I decided to take a class of [this repository](https://github.com/navoshta/traffic-signs/blob/master/Traffic_Signs_Recognition.ipynb).

I adapted the fucntion a little bit in case of function arguments and file handling also the number of maximum images of each class.

The whole prprocess of the image data runs in cell [14]

 * The first step Cell[11] the class provides to flip the      training data in different directions.
 * Second step Cell[8 and 11] is to argument the flipped       data to get a more balanced training set. Why do we       do this step. So like we saw in the histogram above,      the training set is very inhomogenious. That leads to     overfit the model according to some special signs         which are over represented.   

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using from sklearn.model_selection import train_test_spli(). I split it into 25% vilidation and 75% training data. 

My final training set had 113715 number of images. My validation set and test set had 37905 and 12630 number of images. !["Amount and format of train test and validation set"][image4]

Here is an example of an original image and an augmented image:

![alt text][image10]

The difference between the original data set and the augmented data set is the following:

* At first the data after argumentation promises a much better general performance because it has much more images per class. Furthermore it has at least 3000 images per class. So this is a very balanced data set, which prevents to generate a training bias for a special class like the stop sign only because it exists often in the set. Further more the model gets data which is more like images a camera would generate in the real world in example twisted or mirrored.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Locatted in cell[12]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   					| 
| Convolution 5x5     	| 1x1 stride, same valid, outputs 28x28x6   	|
| RELU					| conv1 = tf.nn.relu(conv1)						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Convolution 5x5	    | 1x1 stride, same valid, outputs 10x10x16		|
| RELU					| conv1 = tf.nn.relu(conv1)						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| flatten       	    | Input = 5x5x16 output 400     				|
| Fully connected		| Input = 400. Output = 120						|
| RELU					| conv1 = tf.nn.relu(fc1)						|
| Fully connected		| Input = 400. Output = 84						|
| RELU					| conv1 = tf.nn.relu(fc2)						|
| Fully connected		| Input = 84. Output = 84						|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Located in[28]

To train the model, I used a LeNet model out of the tensor flow lesson from udacity.

The optimizer: Adam optimizer --> I use, cause it is the default optimizer. i did not know much about optimizer advantages or which other optimizer it gives and when should I use that .

The batch size: 128 I like this value. So I experienced a lot with the parameter of the batch size. So and in my opinion this size work well for me.

Number of epoch: I started with 8 epoch and then 10,16,20. As I take the non balanced and argumented training set a number of epoch raound about 8 worked best. So after my trainng set increased up to 113000 this number of epochs did not work very well anymore. So I alternately played with the learning rate and the number of epoch. So in my feeling the best result after a few training sessions was to take a learning rate of 0.0008 and a number of epochs of 30.

So the two other hyperparameter like mu and sigma I did not change. i played a short time but the results were not good.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Located in cell[]

My final model results were:

* validation set accuracy of 0.842  
* test set accuracy of 0.911

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * LeNet: So like described in the project intro I thought it is a good approach to take the LeNet architecture. With this architecture I can prevent many issues and can concentrate on the parameter tuning and special behavior of the model in case of parameter tuning.
* What were some problems with the initial architecture?
    * No problems after adapted it according the classes etc.
* Spending so much effort into get things working in general... aws and on and     on. Furthermore my company notebook was not working as it should and I hab      to buy a new one. Finally I did not have the time for studying different        architecture. Sorry for that.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem?
    * So a convolution makes sense because we can detect 

Dropout is not a good idea like described in [here](https://www.reddit.com/r/MachineLearning/comments/42nnpe/why_do_i_never_see_dropout_applied_in/)

If a well known architecture was chosen:
* What architecture was chosen? no special archtitecture was choosen
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    * So in my opinion there is no evidence that the model working very well in the real world. But as we see we train the model with a lot of images and all images was not perfect. Some of the images was really bad regarding to the contrast and strong shadows. Also the argumenting and flipping ofthe training data is not a evidence but a good and strong hypothesis   which gives a lot of hope to get the model robust for the real world. The argumentation can simulate a little bit the challange of the real world szenarios  
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


[//]: # (Image References)

[image1]: ./examples/exploratyDataSetSinglePrint.PNG "Visualization random training image"
[image2]: ./examples/histogramBalancOfImages.PNG "Plot of how the training data is balanced"
[image3]: ./examples/plotsPreprocessingOutput.png "Random Noise"
[image4]: ./examples/formateAndBalancingAfterPreProcess.png 
[image5]: ./examples/example01.jpg "No entry"
[image6]: ./examples/example02.jpg "Speed limit (30km/h)"
[image7]: ./examples/example03.jpg "Speed limit (80km/h)"
[image8]: ./examples/example04.jpg "Yield"
[image9]: ./examples/example05.jpg "Stop"
[image10]:./examples/originalArgumented.jpg "original and argumented"
