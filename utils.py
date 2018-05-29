import tensorflow as tf
import argparse
import yaml

slim = tf.contrib.slim


def get_init_fn(FLAGS):
    """
    This function is copied from TF slim.

    Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    tf.logging.info('Use pretrained model %s' % FLAGS.loss_model_file)

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)


class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


# def parse_args(yml_path:str, yml_default:str, model_img_path:str):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--conf', default=(yml_path + yml_default), help='yam file path')
#     flags = parser.parse_args() 
#     # flags is a argparse.Namespace object which has params 'conf' 
#     return flags

def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS

def mean_image_subtraction(image, means):
    image = tf.to_float(image)

    num_channels = 3
    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, 2)

