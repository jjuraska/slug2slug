from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate

import yaml
from six import string_types

import tensorflow as tf
from tensorflow import gfile

from seq2seq import tasks, models
from seq2seq.configurable import _maybe_load_yaml, _deep_merge_dict
from seq2seq.data import input_pipeline
from seq2seq.inference import create_inference_graph
from seq2seq.training import utils as training_utils

tf.flags.DEFINE_string("tasks", "{}", "List of inference tasks to run.")
tf.flags.DEFINE_string("model_params", "{}", """Optionally overwrite model
                       parameters for inference""")

tf.flags.DEFINE_string("config_path", None,
                       """Path to a YAML configuration file defining FLAG
                       values and hyperparameters. Refer to the documentation
                       for more details.""")

tf.flags.DEFINE_string("input_pipeline", None,
                       """Defines how input data should be loaded.
                       A YAML string.""")

tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_string("checkpoint_path", None,
                       """Full path to the checkpoint to be loaded. If None,
                       the latest checkpoint in the model dir is used.""")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")

FLAGS = tf.flags.FLAGS

def main(_argv):
    """Program entry point.
    """

    # Load flags from config file
    if FLAGS.config_path:
        with gfile.GFile(FLAGS.config_path) as config_file:
            config_flags = yaml.load(config_file)
            for flag_key, flag_value in config_flags.items():
                setattr(FLAGS, flag_key, flag_value)

    if isinstance(FLAGS.tasks, string_types):
        FLAGS.tasks = _maybe_load_yaml(FLAGS.tasks)

    if isinstance(FLAGS.input_pipeline, string_types):
        FLAGS.input_pipeline = _maybe_load_yaml(FLAGS.input_pipeline)

    input_pipeline_infer = input_pipeline.make_input_pipeline_from_def(
            FLAGS.input_pipeline, mode=tf.contrib.learn.ModeKeys.INFER,
            shuffle=False, num_epochs=1)

    # Load saved training options
    train_options = training_utils.TrainOptions.load(FLAGS.model_dir)

    # Create the model
    model_cls = locate(train_options.model_class) or \
        getattr(models, train_options.model_class)
    model_params = train_options.model_params
    model_params = _deep_merge_dict(
            model_params, _maybe_load_yaml(FLAGS.model_params))
    model = model_cls(
            params=model_params,
            mode=tf.contrib.learn.ModeKeys.INFER)

    # Load inference tasks
    hooks = []
    for tdict in FLAGS.tasks:
        if not "params" in tdict:
            tdict["params"] = {}
        task_cls = locate(tdict["class"]) or getattr(tasks, tdict["class"])
        task = task_cls(tdict["params"])
        hooks.append(task)

    # Create the graph used for inference
    predictions, _, _ = create_inference_graph(
            model=model,
            input_pipeline=input_pipeline_infer,
            batch_size=FLAGS.batch_size)

    saver = tf.train.Saver()
    checkpoint_path = FLAGS.checkpoint_path
    if not checkpoint_path:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        
    
    export_path = 'saved_model'
    print('Exporting trained model to', export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        tf.logging.info('Restored model from ' + checkpoint_path)

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING])

        builder.save()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
