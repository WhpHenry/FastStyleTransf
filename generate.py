import os
import time
import tensorflow as tf
import reader
import model
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. ')
tf.app.flags.DEFINE_string("model_path", "gen_model/", "folder for storing style model")
tf.app.flags.DEFINE_string("model_file", "mosaic.ckpt-done", "storing style model *.ckpt file")
tf.app.flags.DEFINE_string("image_path", "gen_image/", "store input and output images")
tf.app.flags.DEFINE_string("input_image", "test3.jpg", "input image to generate")
tf.app.flags.DEFINE_string("output_image", "res.jpg", "output image from generate")
# tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')

FLAGS = tf.app.flags.FLAGS

def generate():
    height, width = 0, 0
    with open((FLAGS.image_path + FLAGS.input_image), 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.input_image.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            img_prep_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False
            )
            img = reader.get_image((FLAGS.image_path + FLAGS.input_image), 
                                    height, width, img_prep_fn)
            # Add batch dimension
            img = tf.expand_dims(img, 0)
            gen_img = tf.cast(model.img_trans_net(img, training=False), tf.uint8)
            # Remove batch dimension
            gen_img = tf.squeeze(gen_img, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_path + FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            # Make sure FLAGS.image_path directory exists. 
            gen_file = FLAGS.image_path + FLAGS.output_image
            if os.path.exists(FLAGS.image_path) is False:
                os.makedirs(FLAGS.image_path)
            
            # generate image
            with open(gen_file, 'wb') as img:
                start = time.time()
                img.write(sess.run(tf.image.encode_jpeg(gen_img)))
                elapsed = time.time() - start
                print('Elapsed time: {}s'.format(elapsed))
                tf.logging.info('Elapsed time: {}s'.format(elapsed))
                tf.logging.info('Done. Please check {}.'.format(gen_file))

def main(_):
    
    generate()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
