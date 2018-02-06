from __future__ import print_function

import sys
import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
from grpc.beta import implementations
from itertools import repeat

from data_loader import tokenize_mr
from postprocessing import finalize_utterance


tf.app.flags.DEFINE_string('server', '', 'Host:port of the service')
tf.app.flags.DEFINE_string('query', '', 'Input MR')
FLAGS = tf.app.flags.FLAGS

RPC_TIMEOUT = 5.0


def get_utterance_from_mr_bkp(hostport, mr):
    '''
    Sends a request to the prediction service to produce an utterance.
    Args:
        hostport: Host:port address where the service is running.
        mr: THe query MR for which an utterance should be generated.
    Returns:
        The utterance generated from the query MR.
    '''

    # define the connection to the service
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    print('Preprocessing the MR...', end=' ')
    sys.stdout.flush()
    
    # convert the MR into a sequence of tokens
    mr_tokens, mr_dict = tokenize_mr(mr)

    print('Done')
    print('Evaluating the query...', end=' ')
    sys.stdout.flush()
    
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
    predicted_tokens = ' '.join(predicted_tokens[:-1]).strip()
    

    # if we're using beam search we take the first beam
    #if np.ndim(predicted_tokens) > 1:
    #    predicted_tokens = predicted_tokens[:, 0]
    
    print('Done')
    print('Postprocessing the utterance...', end=' ')
    sys.stdout.flush()

    utterance = finalize_utterance(predicted_tokens, mr_dict)

    print('Done')

    return utterance


def get_utterance_from_mr(hostport, mr):
    '''
    Sends a request to the prediction service to produce an utterance.
    Args:
        hostport: Host:port address where the service is running.
        mr: THe query MR for which an utterance should be generated.
    Returns:
        The utterance generated from the query MR.
    '''

    print('Preprocessing the MR...', end=' ')
    sys.stdout.flush()
    
    # convert the MR into a sequence of tokens
    mr_tokens, mr_dict = tokenize_mr(mr)

    print('Done')
    #print('Evaluating the query...', end=' ')
    sys.stdout.flush()
    
    # construct the request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'slug2slug'
    request.model_spec.signature_name = 'predict_utterance'

    request.inputs['mr_tokens'].CopyFrom(tf.contrib.util.make_tensor_proto(mr_tokens, shape=[1, len(mr_tokens)]))
    request.inputs['mr_len'].CopyFrom(tf.contrib.util.make_tensor_proto(len(mr_tokens), shape=[1]))
    
    # send the request
    hostports = ['localhost:9000',
                 'localhost:9001']

    pool = multiprocessing.Pool(len(hostports))
    results = pool.map_async(send_request, zip(hostports, repeat(request)))
    pool.close()
    pool.join()

    utt_candidates = [res.outputs['utterance'].string_val for res in results.get()]

    # DEBUG PRINT
    print(utt_candidates)

    # TODO: beam re-ranking



    predicted_tokens = utt_candidates[0]

    # process the response
    predicted_tokens = ' '.join(predicted_tokens[:-1]).strip()
    

    # if we're using beam search we take the first beam
    #if np.ndim(predicted_tokens) > 1:
    #    predicted_tokens = predicted_tokens[:, 0]
    
    #print('Done')
    print('Postprocessing the utterance...', end=' ')
    sys.stdout.flush()

    utterance = finalize_utterance(predicted_tokens, mr_dict)

    print('Done')

    return utterance


def send_request(task_args):
    hostport, request = task_args

    # define the connection to the service
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    return stub.Predict(request, RPC_TIMEOUT)


def main(_):
    if not FLAGS.server:
        print('Please, specify the server [host:port]')
        return

    if not FLAGS.query:
        print('Please, specify the query MR')
        return

    utterance = get_utterance_from_mr(FLAGS.server, FLAGS.query)
    print(utterance)


if __name__ == '__main__':
    tf.app.run()
