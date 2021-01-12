from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params. Store weights and biases for the convolutional   #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Our network consists of three layers. For the first layer we need to initialize the weights of the filter kernels.
        # For the dimension of W1 we use the following logic. The first dimension must equal the number of filter kernels.
        # The second dimension equals the number of channels of the input images. The third and fourth dimension correspond
        # to the height and width size of the filter kernels. 
        # Furthermore, we must initialize the bias with a dimension which corresponds to the number of filter kernels, since
        # every filter kernel has its own bias value.
        self.params.update({'W1':np.random.normal(0., weight_scale, (num_filters,input_dim[0],filter_size,filter_size))})
        self.params.update({'b1':np.zeros(num_filters)})
        
        # Now we come to the second layer. Since the second layer is a fully connected layer, we need to flatten our image 
        # to a flat feature vector. Since we know that non-overlapping 2x2 pooling is applied to the image before it is 
        # forwarded to the layer, we know that the image will have half the size of the one before max pooling.
        # We also know that the depth of the image will stay the same so we multiply this value with the new dimensions 
        # and get the number of flatten features. We use this as our first dimension of W2. As the second dimension we 
        # need the number of units in the next layer which is provided by the hidden_dim variable.
        # For the bias we just need the number of units in the next layer (hidden_dim).
        flatten_image = int(num_filters * (input_dim[1] / 2) * (input_dim[2] / 2))
        self.params.update({'W2':np.random.normal(0., weight_scale, (flatten_image,hidden_dim))})
        self.params.update({'b2':np.zeros(hidden_dim)})
        
        # Lastly we come to the last fully connected layer. This one maps from the last hidden layer of dimension hidden_dim
        # to the output layer which contains num_classes units.
        # We also initialize num_classes different bias values (for every output class one).
        self.params.update({'W3':np.random.normal(0., weight_scale, (hidden_dim,num_classes))})
        self.params.update({'b3':np.zeros(num_classes)})

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # For the forward pass we can use three simple functions. The first one (conv_relu_pool_forward)
        # takes the input images X and applies convolution using the kernel filters W1 and bias terms b1.
        # It also uses the convolution information stored in conv_param. After that it applies the ReLu 
        # activation function along with the max pooling. For the max pooling it uses the information
        # stored in the pool_param variable. The output of the function is forwarded to the next layer
        # and the cache is stored to support the backprop process.
        # In the next layer we simply use the affine_relu_forward function with the data from the previous 
        # layer and the weights we defined in the task above.
        # In the last step we forward our data one last time through the affine_forward function (since
        # it is the output layer) and return the scores of our network.
        out_layer1, cache_layer1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_layer2, cache_layer2 = affine_relu_forward(out_layer1, W2, b2)
        out_layer3, cache_layer3 = affine_forward(out_layer2, W3, b3)
        scores = out_layer3

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Using the scores and the dataset labels, we can determine the loss and the gradient by 
        # using the softmax_loss function.
        loss, dx = softmax_loss(scores, y)
        
        # Next we need to add the L2 regularization to our loss. For this we just square all our 
        # "learnable" parameters from the three layers (except the bias terms) and sum them together.
        # At the end we multiply this value by our regularization hyperparameter self.reg and 0.5 
        # (as stated by the task). This value is then added to the loss value.
        reg_w1 = np.sum(np.square(W1))
        reg_w2 = np.sum(np.square(W2))
        reg_w3 = np.sum(np.square(W3))
        loss += self.reg * 0.5 * (reg_w1+reg_w2+reg_w3)
        
        # Now we come to the backprop process. For the last layer we use the affine_backward function
        # since we used the affine_forward in the forward process. As parameters we provide it with the
        # gradient from the loss function and the cache values which we stored in the forward process. 
        # Lastly we increase the gradient of w by deriving the gradient of the regularization term with
        # respect to w. We add this new value to our gradient dw.
        dx, dw, db = affine_backward(dx, cache_layer3)
        dw += self.reg*self.params['W3']
        grads.update({'W3': dw})
        grads.update({'b3': db})
        
        # The same explanation from above can be applied here. The only difference is that we are 
        # backpropagating from the second to the first layer and therefore we use the affine_relu_backward
        # function.
        dx, dw, db = affine_relu_backward(dx, cache_layer2)
        dw += self.reg*self.params['W2']
        grads.update({'W2': dw})
        grads.update({'b2': db})
        
        # In the last step we use the conv_relu_pool_backward function since we used the corresponding
        # one in the forward pass. We also calculate once again the derivative of the regularization term.
        dx, dw, db = conv_relu_pool_backward(dx, cache_layer1)
        dw += self.reg*self.params['W1']
        grads.update({'W1': dw})
        grads.update({'b1': db})

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
