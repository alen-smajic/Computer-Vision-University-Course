from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Here we initialize the weights that are located between the input and the hidden layer. For this we add a new key 'W1' to
        # the dictionary and initialize random normal distributet numbers. As parameters we say that the mean is 0 and the standard
        # deviation is weight_scale. Lastly we specify the dimension of our weight matrix. Since it connects from the input layer (size 
        # input_dim) to the hidden layer (hidden_dim) we can see that we need a dimensin of (input_dim, hidden_dim).
        self.params.update({'W1': np.random.normal(0., weight_scale, (input_dim, hidden_dim))})
        # The same applies for 'W2'. The only difference is the shape of the matrix. Since this one connects the hidden layer to the 
        # output we need a dimension of (hidden_dim, num_classes).
        self.params.update({'W2': np.random.normal(0., weight_scale, (hidden_dim, num_classes))})
        
        # The two bias values are initialized as zeros. The size of the vector corresponds to the size of the output we want to 
        # generate. In the case of b1 this is the amount of neurons in the hidden_dim, and in the case of the output layer this is
        # the amount of classes (num_classes).
        self.params.update({'b1': np.zeros(hidden_dim)})
        self.params.update({'b2': np.zeros(num_classes)})

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Here we want to implement the forward pass. Since our hidden layer uses the ReLU function we can use the predefined 
        # function affine_relu_forward. This takes the input X and the weight parameters which map from the input layer
        # to the hidden layer (we defined these in the previous __init__ function). As output we get a new matrix and store
        # the cache values for the backward pass. Next we take this new output matrix and forward it into the affine_forward
        # function (since this is the output layer and we do not apply the ReLU function anymore). We also store the cache
        # values for later. The generated output of form (N,10) stores the scores of every sample for the 10 classes.
        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        out2, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
        scores = out2
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # First we compute the softmax loss based on the scores from the previous section and the labels given in y. For
        # this we use the predefined function softmax_loss. The function also output the gradient of the loss function 
        # with respect to the scores used in the softmax transformation. We can use these to calculate our other gradients
        # in the backward process.
        loss, dout = softmax_loss(scores, y)
        
        # Now we need to add the L2 regularization to this loss. We calculate the squares of every single weight (except
        # the biases, these are not regularized) in our network and sum them all together. Lastly we multiply this scalar
        # value by our regularization hyperparameter self.reg and multiply it with 0.5 as stated by the task. As an output
        # we get a increased loss value (if regularization is used and the self.reg value is non-zero).
        reg_w1 = np.sum(np.square(self.params['W1']))
        reg_w2 = np.sum(np.square(self.params['W2']))
        loss += self.reg * 0.5 * (reg_w1+reg_w2)
        
        # Now we want to calculate our gradients of the weights which connect the hidden and the output layer. To do this
        # we use the affine_backward function (since we used the affine_forward function before) with parameters dout
        # (which we got from the softmax_loss function) and the cache2 (which we stored when we computed the forward pass).
        # The function returns us the gradients of the weights as well as the gradint of the input which we can use for 
        # further backward propagation.
        dx_2, dw_2, db_2 = affine_backward(dout, cache2)
        # We need also to compute the gradient regarding the regularization term, since this one is part of the loss value.
        # When we calculate the derivative we can see that its basically the weights multiplied by the regularization
        # hypterparaneter self_reg. We take these values and add it to the other gradient values to get the correct
        # gradient.
        dw_2 = dw_2 + (self.reg*self.params['W2'])
        grads.update({'W2': dw_2})
        grads.update({'b2': db_2}) 
        
        # Similary to the previous steps we calculate the gradient of the weights which connect the input with the hidden
        # layer. Since we used thee affine_relu_forward function in the forward pass, we need to use the 
        # affine_relu_backward pass to compute the gradients. As input we take the gradients dx_2 from the previous layer
        # and we use the cache1 value which we stored in the forward pass. 
        dx_1, dw_1, db_1 = affine_relu_backward(dx_2, cache1)
        # We also need to computee again the gradient regarding the regularization term. The same explaination from above
        # applys here too.
        dw_1 = dw_1 + (self.reg*self.params['W1'])
        grads.update({'W1': dw_1})
        grads.update({'b1': db_1})  
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # In this task we had to initialize the parameters of the network which can have an arbtirary number of layers. We basically
        # had all informationen which we needed in the form of the num_layers, input_dim, num_classes and hidden_dims variables. The
        # solution isnt any different from the solution in the two layer network. In this case we use a for loop to loop through all 
        # of our possible layers. The improtant aspect here is that every weight W has dimension (previous_layer x next_layer) except the first
        # and the last layer. The first layer has dimension (input_dim x hidden_dims[0]) and the last layer has dimension 
        # (hidden_dims[-1] x num_classes).
        current_dim = input_dim
        for i in range(self.num_layers-1):
                self.params.update({'W'+str(i+1): np.random.normal(0., weight_scale, (current_dim, hidden_dims[i]))})
                self.params.update({'b'+str(i+1): np.zeros(hidden_dims[i])})
                current_dim = hidden_dims[i]
        self.params.update({'W'+str(self.num_layers): np.random.normal(0., weight_scale, (current_dim, num_classes))})
        self.params.update({'b'+str(self.num_layers): np.zeros(num_classes)})
                     
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # For the forward pass we need a way to store the cache values in order to support the backward pass later on.
        # Since we have an arbitrary number of layers and we have to store an unknown amount of cache values, the best
        # way to do this is a dictionary in which we store the information for every operation we are doing.
        caches = {}
        
        # So now we are passing our input forward inside our neural net. This task is also very similar to the two layer
        # network. We are looping through our hidden layers and using the affine_relu_forward function with the input
        # from the previous layer and the parameters we just initialized in the previous task to calculate the output for
        # the next layer. After every pass of affine_relu_forward, we have to make a new dictionary input to store the 
        # cache values. The last step is important. In the last step we have to pass our output from the last hidden layer
        # to our output neurons. However, since this is the last layer, we are not using the affine_relu_forward function
        # anymore but the normal affine_forward. We store the values in our cache dictionary and give the output as our
        # final scores out.
        output = X
        for i in range(self.num_layers-1):
            output, cache = affine_relu_forward(output, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            caches.update({'cache{0}'.format(i+1): cache})
        output, cache = affine_forward(output, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])   
        caches.update({'cache'+str(self.num_layers): cache})
        scores = output
           
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # After we have calculated the scores, we use the softmax_loss function to calculate the
        # corresponding loss and the gradients.
        loss, dout = softmax_loss(scores, y)
        
        # Next we have to add the regularization to our loss. For this we loop through all our 
        # parameters W, square them and sum them together to get one scalar value. In the last 
        # step we multiply this value with the regularization hyperparameter self.reg and with
        # the scalar 0.5 (as statet by the task) and add this new value to the loss.
        reg_w = 0
        for i in range(1, self.num_layers):
            reg_w += np.sum(np.square(self.params['W'+str(i)]))
        loss += self.reg * 0.5 * reg_w
        
        # So now we come to the backward pass. One important aspect to rember is that in the previous
        # task our last layer used the affine_forward function (since it was the last layer). So now
        # in the backward pass, we have to use the affine_backward function to calculate the gradients.
        # As parameters we use the gradient of the softmax function which we just calculated and the
        # cache values from the dictionary. Before we store these values we have also to add the gradient
        # of the regularization term to our result. For this we just multiply the hyperparameter 
        # self.reg with our weight parameters and add the value to the corresponding gradient values.
        dx, dw, db = affine_backward(dout, caches['cache'+str(self.num_layers)])
        
        dw = dw + (self.reg*self.params['W'+str(self.num_layers)])
        
        grads.update({'W'+str(self.num_layers): dw})
        grads.update({'b'+str(self.num_layers): db})
        
        
        for i in range(self.num_layers-1, 0, -1):
            dx, dw, db = affine_relu_backward(dx, caches['cache'+str(i)])
            
            dw = dw + (self.reg*self.params['W'+str(i)])
                                  
            grads.update({'W'+str(i):dw})
            grads.update({'b'+str(i):db})                                       
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
