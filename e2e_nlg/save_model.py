import sys
import tensorflow as tf
from seq2seq import models
from seq2seq import graph_utils
from seq2seq.training import utils as training_utils
from pydoc import locate


tf.app.flags.DEFINE_string('model_dir', '', 'Path to the trained model')
tf.app.flags.DEFINE_string('vocab_dir', '', 'Path to the vocabulary files')
FLAGS = tf.app.flags.FLAGS


def save_model(model_dir, vocab_dir):
    export_path = 'saved_model'
    checkpoint_path = tf.train.latest_checkpoint(model_dir)

    # load saved training options
    train_options = training_utils.TrainOptions.load(model_dir)

    # define tensor inputs to replace the input pipeline
    source_tokens_ph = tf.placeholder(dtype=tf.string, shape=(1, None))
    source_len_ph = tf.placeholder(dtype=tf.int32, shape=(1,))

    print('Rebuilding the model...', end=' ')
    sys.stdout.flush()

    # rebuild the model graph
    model_cls = locate(train_options.model_class) or getattr(models, train_options.model_class)
    model_params = train_options.model_params

    # add beam search parameters
    #model_params['inference.beam_search.beam_width'] = 10
    #model_params['inference.beam_search.length_penalty_weight'] = 1.0

    # DEBUG PRINT
    #print(model_params)

    model = model_cls(params=model_params,
                      mode=tf.contrib.learn.ModeKeys.INFER)

    model(
        features={
            'source_tokens': source_tokens_ph,
            'source_len': source_len_ph
        },
        labels=None,
        params={
            'vocab_source': vocab_dir,
            'vocab_target': vocab_dir
        })

    predictions = graph_utils.get_dict_from_collection('predictions')

    print('Done')
    print('Exporting trained model...')
    sys.stdout.flush()

    saver = tf.train.Saver()
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # build the signature def map
    tensor_info_mr_tokens = tf.saved_model.utils.build_tensor_info(source_tokens_ph)
    tensor_info_mr_len = tf.saved_model.utils.build_tensor_info(source_len_ph)
    tensor_info_utt = tf.saved_model.utils.build_tensor_info(predictions['predicted_tokens'])

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'mr_tokens': tensor_info_mr_tokens,
                    'mr_len': tensor_info_mr_len},
            outputs={'utterance': tensor_info_utt},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    # restore and export the model as SavedModel
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        tf.logging.info('Restored model from ' + checkpoint_path)

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_utterance': prediction_signature
            },
            legacy_init_op=legacy_init_op)

        builder.save()

        print('Model exported to "' + export_path + '"')


def main(_):
    if not FLAGS.model_dir:
        print('Please, specify the path to the trained model you would like to export')
        return

    if not FLAGS.vocab_dir:
        print('Please, specify the path to the vocabulary files of the model')
        return

    save_model(FLAGS.model_dir, FLAGS.vocab_dir)


if __name__ == "__main__":
    tf.app.run()
