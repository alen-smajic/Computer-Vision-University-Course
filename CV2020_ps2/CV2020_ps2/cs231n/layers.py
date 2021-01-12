from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # In this line of code we take our input x which has the shape (N, d_1, ..., d_k) and transform it to the shape (N,D) where 
    # D = d_1 * ... * d_k. 
    new_x = np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    
    # Now we take our reshaped input and multiply it with the given feature weights. Input has shape (N,D), the weights have
    # shape (D,M) so the output of this multiplication has shape (N,M) as desired.
    out = np.matmul(new_x, w)

    # Lasty we add the bias vector of shape (M,) to every single row of our output matrix (N, M). 
    out += b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Using the chain rule we can derive that the gradient with respect to x is the matrix multiplication of dout and w. To do this, we 
    # have to transpose our w, so we can get (N, M) * (M, D) = (N, D). Lastly we reshape the dx from (N, D) to (N, d_1, ..., 
    # d_k) as stated by the task.
    dx = np.reshape(np.matmul(dout, w.T), x.shape)
    
    # Next we want to calculate our dw. Using again the chain rule we can derive that the gradient with respect to w is the matrix mult. of
    # dout and x. To do this, we first have to reshape our x as in the affine_forward function to get it to the 
    # form (N, D). Since our output dw has to be in the shape (D, M), we need to transpose our new_x to get the desired results. 
    # At the end we get the following shapes (D, N) * (N, M) = (D, M).
    new_x = np.reshape(x,(x.shape[0], np.prod(x.shape[1:])))
    dw = np.matmul(new_x.T, dout)
    
    # Our last task was to calculate db. However this task is not that trivial. When we calculate our forward pass, upon summing
    # the bias on our output, our bias is actually composed like this (N,1)*(1,M) where (N,1) is a vector which consists only 
    # of ones and (1,M) is our given bias term (transposed). This produces a matrix of shape (N,M) which we can add to the output
    # of the first multiplication (has also dimension (N,M)). When we now want to calculate our derivate this comes in handy. Using the chain rule we can derive
    # the gradient with respect to b as the matrix mult. of dout and this vector (N,1) which contains only ones. Since our result needs to
    # have the shape (M,) we have to transpose our vector containing ones. At the end we get the following result: 
    # (1,N)*(N,M)=(1,M) and by transposing it we get the desired (M,).
    db = np.matmul(np.ones(x.shape[0]).T, dout).T
       
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # The ReLu retains the positive values and floors the negative values to zero.
    out = np.maximum(x, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # The gradient with respect to x is: dout element-wise multiplied with the gradient of the ReLu function with respect to x. 
    # This latter gradient is 1 if the value is > 0 and 0 if the value is <= 0. So basically the code below is cecking for which indexes of 
    # x the values negative are so it can floor the corresponding value in dout to 0. The other values are basically retained.
    dx = np.multiply(dout, (x>0)*1)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # To generate the dropout mask, we first need to initialize a matrix with random values between 0 and 1 with the same shape as our input x.
        # Now we compare every value of this matrix with our dropout rate p. If the value is smaller we keep it. However if its bigger we drop it.
        # Since the values are initialized randomly, we have a probability of p that a value will be kept. After this operation we get a matrix
        # filled with True and False. Since we are implementing the inverted dropout, we need to element-wise divide every value by p. At the end
        # our matrix has either values 1/p or zero-values. We then proceed to element-wise multiply this mask with our input.        
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # In the test mode there is no dropout applied. Since we implemented the inverted dropout we can just return the input as it is. 
        # If we havent implement the inverted dropout, we would have multiplied our input x with p for the test mode.        
        out = x 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Using the chain rule, we can derive that the gradient with respect to x is dout element-wise multiplied with the gradient of the 
        # dropout function with respect to x. Since the dropout function is simply x*mask we can derive that the gradient is simply the mask.
        # So to calculate dx we just have to multiply dout with mask.   
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # First we initialize our output matrix with the correct dimensions. The first dimension equals the number of samples and the
    # second dimension equals the number of filter kernels. For the third and fourth dimension (height and width) we need to use
    # a formula to calculate it because it depends on the padding, the stride, the input height and width dimensions as well as 
    # the filter kernel height and width dimensions. The formula was given above in the comments.
    out_H = int(1 + (x.shape[2] + 2 * conv_param['pad'] - w.shape[2]) / conv_param['stride'])
    out_W = int(1 + (x.shape[3] + 2 * conv_param['pad'] - w.shape[3]) / conv_param['stride'])
    out = np.zeros((x.shape[0], w.shape[0], out_H, out_W))
    
    # Next we add padding to our input. For this we can use the np.pad function as suggested by the hint. We provide it with the
    # padding value and apply the padding only to the third and fourth dimension of the input (height and width).
    x_padding = np.pad(x, pad_width=((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad'])))
    
    # Now we come to the convolution. This first block which consists of 4 for loops is used to loop through all the dimension
    # of the output matrix. It basically gives us access to every single point inside the output matrix, which we need in order
    # to fill it with the correct output values. We also use it to move across the recceptive field.
    for image_count in range(out.shape[0]):
        for filters in range(out.shape[1]):
            for out_h in range(out.shape[2]):
                for out_w in range(out.shape[3]):
                    # The second block of for loops is used to loop through all dimension of the filter kernel. It gives us access
                    # to every single point inside the filter kernel, which we need in order to multiply the value with the 
                    # corresponding input value (the receptive field).
                    for kernel_channels in range(w.shape[1]):
                        for filter_h in range(w.shape[2]):
                            for filter_w in range(w.shape[3]):
                                # In the last step we use the indexes provided by the for loops to slide the filter across the images
                                # with respect to the stride. By doing so we are calculating the convolution and storing it in our
                                # output matrix.
                                out[image_count, filters, out_h, out_w] += w[filters, kernel_channels, filter_h, filter_w] * x_padding[image_count, kernel_channels,                                 out_h*conv_param['stride'] + filter_h, out_w*conv_param['stride'] + filter_w]
            # Once we are done with the convolution using a single filter, we have to add the bias value to every value of our ouput matrix.
            out[image_count, filters] += b[filters]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # We "unpack" the cache values we stored during the forward convolution, in order to use them for the backprop.
    x, w, b, conv_param = cache

    # We are adding padding to our input x, since we need the corresponding dimension values in order to calculate the
    # backprop gradient values for dx.
    x_padding = np.pad(x, pad_width=((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad'])))
    
    # Here we initialize the dimension of our three gradients we want to calculate. Notice that we do not initialize dx with the 
    # normal dimensions of x but rather with the padded version of x. This is done in order to support the backprop convolution 
    # process. We will unpad it in the latter process to return the correct dimensions.
    dx = np.zeros((x_padding.shape))
    dw = np.zeros((w.shape))
    db = np.zeros((b.shape))

    # Calculation of dx: Since the filter kernels are applied on our input x to generate the output values, we need to use them on 
    # our upstream gradients dout in order to calculate dx. We are looping through every single value of our dout using the 4 for 
    # loops. We know that in the forward process, every value of out was calculated by overlaying the filter kernel over the input
    # x and multiplying the values. For the backprop process we need to multiply now this dout value with the filter kernel and 
    # overlay the output of this multiplication over the receptive field in x. By doing this for all the values of dout we are basically 
    # gradually calculating the gradients dx.
    #
    # Calculation of dw: The calculation of dw is similar to that of dx. The difference now is that we have to use the input x 
    # insted of the filter kernels w. We loop through every single value of dout and multiply it with all the values of x, which
    # were used to generate the corresponding output value by overlaying the filter (the receptive field). To support this operation, we have to
    # determine the receptive field of x which was used to overlay the kernel filter and to produce the corresponding output value. This 
    # is done through the variable x_overlay. Once we have determined the region, we just multiply it with the corresponding 
    # dout value and store it as the gradient of this particular filter. This is then repeated for all images, gradually increasing
    # the gradient till we get the correct values for the whole batch. 
    #
    # Calculation of db: In the forward convolution, after the filter kernels are slided over the input image the last step was
    # to add the bias value to all the output values of the respective filter (every filter has its own bias value). Since the bias 
    # influenced every single value of the output, now for the backprop we need to sum all the values dout of a filter dimension
    # to produce the gradients with respect to the bias. We do this for every single image of our batch, gradually increasing
    # our gradient for every single filter. 
    for images in range(dout.shape[0]):
        for filters in range(dout.shape[1]):
            for height in range(dout.shape[2]):
                for width in range(dout.shape[3]):
                    # dx
                    dx[images, :, (height*conv_param['stride']):((height*conv_param['stride'])+w.shape[2]), (width*conv_param['stride']):((width*conv_param['stride'])+w.shape[3])] += w[filters] * dout[images, filters, height, width]
                    
                    # dw
                    x_overlay = x_padding[images, :, (height*conv_param['stride']):((height*conv_param['stride'])+w.shape[2]), (width*conv_param['stride']):((width*conv_param['stride'])+w.shape[3])]
                    dw[filters] += x_overlay * dout[images, filters, height, width]
                    
                    # db
                    db[filters] += dout[images,filters,height,width]

    # After we are done with the backpropagation of the gradient we need to unpad our dx to return it in the correct shape.                
    dx = dx[:, :, conv_param['pad']:(conv_param['pad'] + x.shape[2]), conv_param['pad']:(conv_param['pad'] + x.shape[3])]
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # First we initialize our output matrix with the correct dimensions. The first dimension equals the number of samples and the
    # second dimension equals the number of channels. For the third and fourth dimension (height and width) we need to use
    # a formula to calculate it because it depends on the the stride, the pool height and pool width dimensions as well as the
    # input height and width dimensions. The formula was given in the comments above.
    out_H = int(1 + (x.shape[2] - pool_param['pool_height']) / pool_param['stride'])
    out_W = int(1 + (x.shape[3] - pool_param['pool_width']) / pool_param['stride'])    
    out = np.zeros((x.shape[0], x.shape[1], out_H, out_W))
    
    # We loop through all dimensions of our output which we defined above. By doing so we are sliding across the input x and 
    # returning the maximum value of the receptive field which is defined by the pool height, width and stride value. 
    for images in range(out.shape[0]):
        for channels in range(out.shape[1]):
            for height in range(out.shape[2]):
                for width in range(out.shape[3]):
                    out[images,channels,height,width] = np.max(x[images,channels,(height*pool_param['stride']):((height*pool_param['stride'])+pool_param['pool_height']),(width*pool_param['stride']):((width*pool_param['stride'])+pool_param['pool_width'])])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # We unpack the cache values which we stored during forward pooling.
    x, pool_param = cache
    
    # We initialize dx with zeros using the same shape as our input x.
    dx = np.zeros((x.shape))
    
    # For the backprop of the max pool operation we basically repeat the same process as in the max pooling forward
    # process. The only difference is that we are not returning the max value of the particular area (receptive field), but rather
    # the coordinates where the max value was stored. We need those coordinates because in the backprop of the 
    # max pool function, the gradient is only backpropagated to the values which contributed to max pooling as
    # max values. Once we have those coordinates we can simply update those values in dx with the corresponding
    # upper stream derivative dout. All other values from the receptive field in dx are floored to zero. 
    for images in range(dout.shape[0]):
        for channels in range(dout.shape[1]):
            for height in range(dout.shape[2]):
                for width in range(dout.shape[3]):
                    cords =  np.where(x == np.amax(x[images,channels,(height*pool_param['stride']):((height*pool_param['stride'])+pool_param['pool_height']),(width*pool_param['stride']):((width*pool_param['stride'])+pool_param['pool_width'])]))
                    dx[cords] = dout[images,channels,height,width]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
