import tensorflow as tf 

def conv2d(x, ipt_filters, opt_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, ipt_filters, opt_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        # print('** {}'.format(mode))

        padding = tf.pad(x, [[0, 0], [int(kernel/2), int(kernel/2)], 
                             [int(kernel/2), int(kernel/2)], [0, 0]], mode=mode)
        return tf.nn.conv2d(padding, weight, 
                            strides=[1, strides, strides, 1], 
                            padding='VALID', name='conv') 


def conv2d_transpose(x, ipt_filters, opt_filters, kernel, strides):
    with tf.variable_scope('conv_transpose'):
        shape = [kernel, kernel, opt_filters, ipt_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        opt_shape = tf.stack([batch_size, height, width, opt_filters])
        return tf.nn.conv2d_transpose(x, weight, opt_shape, 
                                      strides=[1, strides, strides, 1], name='conv_transpose')

def conv2d_resize(x, ipt_filters, opt_filters, kernel, strides, training):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_h = height * strides * 2
        new_w = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_h, new_w], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return conv2d(x_resized, ipt_filters, opt_filters, kernel, strides)

def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def batch_norm(x, size, training, decay=0.999):
    # insteaded by instance normalize
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    pop_mean = tf.Variable(tf.zeros([size]))
    pop_var = tf.Variable(tf.ones([size]))
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1-decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1-decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, 
                                             beta, scale, epsilon, name='batch_nrom')
    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, 
                                         beta, scale, epsilon, name='batch_norm')

    return tf.cond(training, batch_statistics, population_statistics)

def relu(x):
    r = tf.nn.relu(x)
    # convert nan in r to zero
    return tf.where(tf.equal(r, r), r, tf.zeros_like(r))

def residual(x, filters, kernel, strides):
    with tf.variable_scope('residual'):
        cov1 = conv2d(x, filters, filters, kernel, strides)
        cov2 = conv2d(relu(cov1), filters, filters, kernel, strides)
        return x + cov2

def img_trans_net(img, training):
    # padding for less border
    img = tf.pad(img, [[0, 0], [10, 10], [10, 10], [0, 0]], mode = 'REFLECT')

    with tf.variable_scope('conv1'):
        conv1 = relu(instance_norm(conv2d(img, 3, 32, 9, 1)))
    with tf.variable_scope('conv2'):
        conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))
    with tf.variable_scope('conv3'):
        conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    with tf.variable_scope('deconv1'):
        deconv1 = relu(instance_norm(conv2d_resize(res5, 128, 64, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        deconv2 = relu(instance_norm(conv2d_resize(deconv1, 64, 32, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    # Add 1 and mut 127.5 effect the contrast ratio of result image
    y = (deconv3 + 1) * 127.5   

    # remove padding
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

    return y
