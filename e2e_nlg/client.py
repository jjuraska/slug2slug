"""A client that talks to tensorflow_model_server loaded with mnist model.
The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.
Typical usage example:
        mnist_client.py --server=localhost:9000
"""

import sys

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/home/juriq/Git/seq2seq/e2e_nlg/saved_model', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

RPC_TIMEOUT = 5.0


def do_inference(hostport, work_dir):
    '''
    Tests PredictionService with concurrent requests.
    Args:
        hostport: Host:port address of the PredictionService.
        work_dir: The full path of working directory for test data set.
    Returns:
        The utterance generated from the input MR.
    Raises:
        IOError: An error occurred processing test data set.
    '''

    # define the connection to the service
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    
    # get the input
    mr = 'name &slot_vow_name& area city centre familyfriendly no'
    label = 'There is a place in the city centre, Alimentum, that is not family-friendly.'

    mr_tokens = mr.split() + ['SEQUENCE_END']
    
    # construct the request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'slug2slug'
    request.model_spec.signature_name = 'predict_utterance'

    request.inputs['mr_tokens'].CopyFrom(tf.contrib.util.make_tensor_proto(mr_tokens, shape=[1, len(mr_tokens)]))
    request.inputs['mr_len'].CopyFrom(tf.contrib.util.make_tensor_proto(len(mr_tokens), shape=[1]))
    
    # send the request
    result = stub.Predict(request, RPC_TIMEOUT)

    # process the response
    predicted_tokens = result.outputs['utterance'].string_val
    #predicted_tokens = np.char.decode(predicted_tokens.astype('S'), 'utf-8')
    predicted_tokens = ' '.join(predicted_tokens).split('SEQUENCE_END')[0].strip()

    # if we're using beam search we take the first beam
    if np.ndim(predicted_tokens) > 1:
        predicted_tokens = predicted_tokens[:, 0]

    return predicted_tokens, label


def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return

    utterance, label = do_inference(FLAGS.server, FLAGS.work_dir)
    print('\nGenerated utterance:', utterance)
    print('\nReference utterance:', label)


if __name__ == '__main__':
    tf.app.run()
