The environment of this code:
Python 3.5
CUDA 8.0
tensorflow 1.2

	Please run this code to tennerize the output of the neural networks
1.Please make sure you have get the original parameters and copied it into the file:net_data/le_net
2.You can run the leNet.py to tennerize the neural networks and train them and increace the batchsize to make it perform better, and the parameters will be store in the file:net_data/le_net + "the accuracy of the parameter"(I have get the parameters)
3.After getting the parameters you can run the leNet_test.py to know the accuracy of this set of parameters witch has been binarized and trained.
4.You can import the file ops.py to cahnge the activation function into leaky-ReLU, and the file batch.py to add batch normalization.