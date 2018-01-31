from pydoc import locate
import tensorflow as tf
import numpy as np
from seq2seq import tasks, models
from seq2seq.training import utils as training_utils
from seq2seq.tasks.inference_task import InferenceTask, unbatch_dict

from seq2seq import graph_utils


class DecodeOnce(InferenceTask):
  '''
  Similar to tasks.DecodeText, but for one input only.
  Source fed via features.source_tokens and features.source_len
  '''
  def __init__(self, params, callback_func):
    super(DecodeOnce, self).__init__(params)
    self.callback_func=callback_func
  
  @staticmethod
  def default_params():
    return {}

  def before_run(self, _run_context):
    fetches = {}
    fetches['predicted_tokens'] = self._predictions['predicted_tokens']
    fetches['features.source_tokens'] = self._predictions['features.source_tokens']
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, _run_context, run_values):
    fetches_batch = run_values.results
    for fetches in unbatch_dict(fetches_batch):
      # Convert to unicode
      fetches['predicted_tokens'] = np.char.decode(
          fetches['predicted_tokens'].astype('S'), 'utf-8')
      predicted_tokens = fetches['predicted_tokens']

      # If we're using beam search we take the first beam
      # TODO: beam search top k
      if np.ndim(predicted_tokens) > 1:
        predicted_tokens = predicted_tokens[:, 0]

      fetches['features.source_tokens'] = np.char.decode(
          fetches['features.source_tokens'].astype('S'), 'utf-8')
      source_tokens = fetches['features.source_tokens']
      
      self.callback_func(source_tokens, predicted_tokens)


# TODO: pass via args
MODEL_DIR = 'model'
checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)

# Load saved training options
train_options = training_utils.TrainOptions.load(MODEL_DIR)

# Create the model
model_cls = locate(train_options.model_class) or \
    getattr(models, train_options.model_class)
model_params = train_options.model_params

model = model_cls(params=model_params,
                  mode=tf.contrib.learn.ModeKeys.INFER)


# first dim is batch size
source_tokens_ph = tf.placeholder(dtype=tf.string, shape=(1, None))
source_len_ph = tf.placeholder(dtype=tf.int32, shape=(1,))

model(
    features={
        'source_tokens': source_tokens_ph,
        'source_len': source_len_ph
    },
    labels=None,
    params={
        'vocab_source': 'data',
        'vocab_target': 'data'
    })

saver = tf.train.Saver()

#def _session_init_op(_scaffold, sess):
#    saver.restore(sess, checkpoint_path)
#    tf.logging.info('Restored model from %s', checkpoint_path)

#scaffold = tf.train.Scaffold(init_fn=_session_init_op)
#session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)


#def _tokens_to_str(tokens):
#    return ' '.join(tokens).split('SEQUENCE_END')[0].strip()

## A hacky way to retrieve prediction result from the task hook...
#prediction_dict = {}
#def _save_prediction_to_dict(source_tokens, predicted_tokens):
#    prediction_dict[_tokens_to_str(source_tokens)] = _tokens_to_str(predicted_tokens)

#sess = tf.train.MonitoredSession(
#    session_creator=session_creator,
#    hooks=[DecodeOnce({}, callback_func=_save_prediction_to_dict)])


export_path = 'saved_model'
print('Exporting trained model to', export_path)

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

predictions = graph_utils.get_dict_from_collection('predictions')

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


# The main API exposed
def query_once(source_tokens):
    tf.reset_default_graph()
    source_tokens = source_tokens.split() + ['SEQUENCE_END']
    sess.run([], {
        source_tokens_ph: [source_tokens],
        source_len_ph: [len(source_tokens)]
    })
    return prediction_dict.pop(_tokens_to_str(source_tokens))

      
if __name__ == '__main__':
    samples = [
        'name &slot_vow_name& area city centre familyfriendly no',
        'name &slot_con_name& eattype coffee shop food chinese pricerange £20-25 customer rating high area city centre familyfriendly no near &slot_con_near&',
        'name &slot_con_name& eattype coffee shop food english pricerange moderate customer rating 1 out of 5 near &slot_con_near&'
    ]
    for sample_in in samples:
        print(sample_in)
        #print(query_once(sample_in))
        print()
  