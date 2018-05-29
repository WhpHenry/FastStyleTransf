
import os
import time
import tensorflow as tf

import reader
import model
import loss_func as loss
from utils import get_init_fn, read_conf_file
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

_LOG_EPS = 10
_SUMMARY_EPS = 25
_SAVE_EPS = 1000
_YML_PATH = 'conf/'
_DEFAULT_YML = 'mosaic.yml'
_TRAIN_IMG_PATH = 'imgs4train/'


def training(FLAGS):
    # FLAGS from *.yml file, reading by utils.read_conf_file
    # ensure training path exists
    # in *yml: <model_path>/<naming> models/f.yml
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming) 
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # # create training network
            net_fn = nets_factory.get_network_fn(
                FLAGS.loss_model,
                num_classes=1,
                is_training=False
            )
            img_prep_fn, img_unpr_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False
            )
            ipt_imgs = reader.image(
                FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                _TRAIN_IMG_PATH, img_prep_fn, epochs=FLAGS.epoch 
            )
            # # image transfrom network
            imgs_net = model.img_trans_net(ipt_imgs, training=True)
            opt_imgs = [img_prep_fn(img, FLAGS.image_size, FLAGS.image_size)
                        for img in tf.unstack(imgs_net, axis=0, num=FLAGS.batch_size)]
            _, endpoints_d = net_fn(tf.concat([opt_imgs, ipt_imgs], 0), spatial_squeeze=False)

             # Log the structure of loss network
            tf.logging.info('Loss network layers(content_layers and style_layers):')
            for key in endpoints_d:
                tf.logging.info(key)
            
            # # loss network
            L_content = loss.content_loss(endpoints_d, FLAGS.content_layers)
            style_features = loss.get_style_feature(FLAGS)
            L_style, L_style_sum = loss.style_loss(endpoints_d, style_features, FLAGS.style_layers)
            L_tv = loss.total_loss(opt_imgs)
            # weight defined in conf/*.yml
            # in mosaic.yml, W(content, style, total variation loss) = (1, 100, 0) 
            l = FLAGS.style_weight*L_style + FLAGS.content_weight*L_content + FLAGS.tv_weight*L_tv

            # Add Summary for visualization in tensorboard.
            tf.summary.scalar('loss/content_loss', L_content)
            tf.summary.scalar('loss/style_loss', L_style)
            tf.summary.scalar('loss/regularizer_loss', L_tv)

            tf.summary.scalar('weighted_loss/weighted_content_loss', L_content * FLAGS.content_weight)
            tf.summary.scalar('weighted_loss/weighted_style_loss', L_style * FLAGS.style_weight)
            tf.summary.scalar('weighted_loss/weighted_regularizer_loss', L_tv * FLAGS.tv_weight)
            tf.summary.scalar('total_loss', l)

            for layer in FLAGS.style_layers:
                tf.summary.scalar('style_loss/' + layer, L_style_sum[layer])
            tf.summary.image('generated img transf net', imgs_net)
            # tf.image_summary('processed_generated', processed_generated)  # May be better?
            tf.summary.image('origin', tf.stack([
                img_unpr_fn(img) for img in tf.unstack(ipt_imgs, axis=0, num=FLAGS.batch_size)
            ]))
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path)  
            
            # # prepare for training
            global_step = tf.Variable(0, name='global_step', trainable=False)

            var2train = []
            for var in tf.trainable_variables():
                if not var.name.startswith(FLAGS.loss_model):
                    var2train.append(var)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(l, global_step=global_step, var_list=var2train)          

            var2restore = []
            for v in tf.global_variables():
                if not v.name.startswith(FLAGS.loss_model):
                    var2restore.append(v)
            saver = tf.train.Saver(var2restore, write_version=tf.train.SaverDef.V2)

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # Restore variables for loss network.
            init_func = get_init_fn(FLAGS)
            init_func(sess)

            # Restore variables for training model if the checkpoint file exists.
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            # # start training
            coord = tf.train.Coordinator()  # manage threads
            threads = tf.train.start_queue_runners(coord=coord)
            start = time.time()
            try:
                while not coord.should_stop():
                    _, loss_t, step = sess.run([train_op, l, global_step])
                    elapsed = time.time() - start
                    start = time.time()
                    # logging
                    if step % _LOG_EPS == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, elapsed))
                    # summary
                    if step % _SUMMARY_EPS == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    # checkpoint
                    if step % _SAVE_EPS == 0:
                        saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt-done'))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

def main(flag):
    training(flag)

if __name__ == '__main__':
    FLAGS = read_conf_file(_YML_PATH + _DEFAULT_YML)
    main(FLAGS)
        



