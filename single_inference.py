# Code partially adopted from https://gist.github.com/noname01/e91e9bea678f0b2f1c9bd283f5b0f452

import sys
import numpy as np
import tensorflow as tf
from seq2seq import tasks, models
from seq2seq.training import utils as training_utils
from seq2seq.tasks.inference_task import InferenceTask, unbatch_dict
from pydoc import locate


tf.app.flags.DEFINE_string('model_dir', '', 'Path to the trained model')
tf.app.flags.DEFINE_string('vocab_dir', '', 'Path to the vocabulary files')
FLAGS = tf.app.flags.FLAGS


prediction_dict = {}


class DecodeOnce(InferenceTask):
    '''
    Similar to tasks.DecodeText, but for a single input only.
    Source fed via features.source_tokens and features.source_len.
    '''

    def __init__(self, params, callback_func):
        super(DecodeOnce, self).__init__(params)
        self.callback_func = callback_func

    @staticmethod
    def default_params():
        return {}

    def before_run(self, _run_context):
        fetches = {
            'predicted_tokens': self._predictions['predicted_tokens'],
            'features.source_tokens': self._predictions['features.source_tokens']
        }

        return tf.train.SessionRunArgs(fetches)

    def after_run(self, _run_context, run_values):
        fetches_batch = run_values.results

        for fetches in unbatch_dict(fetches_batch):
            fetches['features.source_tokens'] = np.char.decode(fetches['features.source_tokens'].astype('S'), 'utf-8')
            source_tokens = fetches['features.source_tokens']

            fetches['predicted_tokens'] = np.char.decode(fetches['predicted_tokens'].astype('S'), 'utf-8')
            predicted_tokens = fetches['predicted_tokens']

            # if using beam search, take the first beam
            if predicted_tokens.shape[1] > 1:
                predicted_tokens = predicted_tokens[:, 0]

            self.callback_func(source_tokens, predicted_tokens)


def query_once(sess, source_tokens_ph, source_len_ph, source_tokens):
    '''
    Performs a single inference for the given MR (sequence of tokens).

    :param sess: TensorFlow session for the restored model.
    :param source_tokens_ph: TensorFlow placeholder for the tokens input.
    :param source_len_ph: TensorFlow placeholder for the tokens input length.
    :param source_tokens: Sequence of (delexicalized) tokens representing the query MR.
    :return: The predicted utterance as a string.
    '''

    tf.reset_default_graph()
    source_tokens = source_tokens.split() + ['SEQUENCE_END']

    sess.run([], {
        source_tokens_ph: [source_tokens],
        source_len_ph: [len(source_tokens)]
    })

    return prediction_dict.pop(_tokens_to_str(source_tokens))


def restore_model(model_dir, vocab_dir):
    '''
    Restores a trained seq2seq model.

    :param model_dir: Path to the directory containing the trained seq2seq model.
    :param vocab_dir: Path to the directory containing the vocabulary files of the model.
    :return: TensorFlow session for the restored model, and two TensorFlow input placeholders.
    '''

    checkpoint_path = tf.train.latest_checkpoint(model_dir)

    # load saved training options
    train_options = training_utils.TrainOptions.load(model_dir)

    # define tensor inputs to replace the input pipeline
    source_tokens_ph = tf.placeholder(dtype=tf.string, shape=(1, None))
    source_len_ph = tf.placeholder(dtype=tf.int32, shape=(1,))

    # rebuild the model graph
    model_cls = locate(train_options.model_class) or getattr(models, train_options.model_class)
    model_params = train_options.model_params

    # add beam search parameters
    model_params['inference.beam_search.beam_width'] = 10
    model_params['inference.beam_search.length_penalty_weight'] = 0.6

    # DEBUG PRINT
    # print(model_params)

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

    saver = tf.train.Saver()

    def _session_init_op(_scaffold, sess):
        saver.restore(sess, checkpoint_path)
        tf.logging.info('Restored model from %s', checkpoint_path)

    scaffold = tf.train.Scaffold(init_fn=_session_init_op)
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)

    sess = tf.train.MonitoredSession(
        session_creator=session_creator,
        hooks=[DecodeOnce({}, callback_func=_save_prediction_to_dict)])

    return sess, source_tokens_ph, source_len_ph


def _tokens_to_str(tokens):
    return ' '.join(tokens).split('SEQUENCE_END')[0].strip()


# retrieve prediction result from the task hook
def _save_prediction_to_dict(source_tokens, predicted_tokens):
    prediction_dict[_tokens_to_str(source_tokens)] = _tokens_to_str(predicted_tokens)


def main(_):
    sample_mrs = [
        'name &slot_vow_name& area city centre familyfriendly no',
        'name &slot_con_name& eattype coffee shop food &slot_con_cuisine_food& pricerange £20-25 customer rating high area city centre familyfriendly no near &slot_con_near&',
        'name &slot_con_name& eattype coffee shop food &slot_vow_cuisine_food& pricerange moderate customer rating 1 out of 5 near &slot_con_near&'
    ]

    if not FLAGS.model_dir:
        print('Error: Please, specify the path to the directory containing the trained model you would like to use.')
        return

    if not FLAGS.vocab_dir:
        print('Error: Please, specify the path to the directory containing the vocabulary files of the model.')
        return

    print('Restoring the model...')
    sys.stdout.flush()

    sess, source_tokens_ph, source_len_ph = restore_model(FLAGS.model_dir, FLAGS.vocab_dir)

    print('\nPredicting...\n')
    sys.stdout.flush()

    # run a test with sample MRs
    for mr in sample_mrs:
        print(mr)
        print(query_once(sess, source_tokens_ph, source_len_ph, mr))
        print()


if __name__ == '__main__':
    tf.app.run()
