
import os
import utils
import tensorflow as tf

from nets import nets_factory
from preprocessing import preprocessing_factory

# folder for saving style images
# during training, when get style feature
# it will  get style feature and generate target style image,
# it will be stored here as target_style_naming.jpg
_MODEL_IMG_PATH = 'gen_model/model_img/'  

# total variation loss
def total_loss(layer):
    shape  = tf.shape(layer)
    h = shape[1]
    w = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.stack([-1,h-1,-1,-1])) - \
        tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.stack([-1,-1,w-1,-1])) - \
        tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# gram matrix: style mapping matrix
# extract stype of img
def gram(layer):
    shape = tf.shape(layer)
    c = shape[0]        # count of images
    w = shape[1]
    h = shape[2]
    f = shape[3]        # count of filters
    fai = tf.reshape(layer, tf.stack([c, -1, f]))   # function fai in lecture
    return tf.matmul(fai, fai, adjoint_a=True) / tf.to_float(w * h * f)

def style_loss(endpoint_d, style_fea_t, style_layers):
    _style_loss = 0
    style_loss_sum = {}
    for style_gram, layer in zip(style_fea_t, style_layers):
        gen_img, _ = tf.split(endpoint_d[layer], 2, 0)
        size = tf.size(gen_img)
        style_loss_i = tf.nn.l2_loss(gram(gen_img) - style_gram) * 2 / tf.to_float(size)
        style_loss_sum[layer] = style_loss_i
        _style_loss += style_loss_i
    return _style_loss, style_loss_sum

def content_loss(endpoint_d, content_layers):
    con_loss = 0    
    for l in content_layers:
        gen_img, tar_img = tf.split(endpoint_d[l], 2, 0)
        size = tf.size(gen_img)
        con_loss += tf.nn.l2_loss(gen_img - tar_img) / tf.to_float(size)
    return con_loss

def get_style_feature(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        net_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes = 1,
            is_training = False
        )
        img_pre_fn, img_unpr_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False
        ) 

        # get image style
        size = FLAGS.image_size
        imgb = tf.read_file(FLAGS.style_image)
        if FLAGS.style_image.lower().endswith('.png'):
            img = tf.image.decode_png(imgb)
        else:
            img = tf.image.decode_jpeg(imgb)
        # add the batch dimension
        imgs = tf.expand_dims(img_pre_fn(img, size, size), 0)

        _, endpoint_d = net_fn(imgs, spatial_squeeze=False)
        # store feature in each layer from endpoint_d[layer]
        features = []
        for layer in FLAGS.style_layers:
            feature = endpoint_d[layer]
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
            features.append(feature)
        
        with tf.Session() as sess:
            init_func = utils.get_init_fn(FLAGS)
            init_func(sess)

            if os.path.exists(_MODEL_IMG_PATH) is False:
                os.makedirs(_MODEL_IMG_PATH)
            savef = _MODEL_IMG_PATH + 'target_style_' + FLAGS.naming + '.jpg'
            with open(savef, 'wb') as wf:
                # remove the batch dimension
                target_img = img_unpr_fn(imgs[0, :])
                val = tf.image.encode_jpeg(tf.cast(target_img, tf.uint8))
                wf.write(sess.run(val))
                tf.logging.info('Target style pattern is saved to: %s.' % savef)

            # return style features from loss net
            # in lecture there should be 4 style opt layer in Fig.2
            return sess.run(features)

