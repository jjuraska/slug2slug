from __future__ import print_function

import sys
import itertools
from operator import itemgetter
import multiprocessing
import time
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
from grpc.beta import implementations, interfaces
from grpc.framework.interfaces.face import face
from itertools import repeat
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
from tensor2tensor.data_generators import text_encoder

from data_loader import tokenize_mr
from slot_alignment import scoreAlignment
from postprocessing import finalize_utterance


tf.app.flags.DEFINE_string('query', '', 'Input MR')
FLAGS = tf.app.flags.FLAGS

RPC_TIMEOUT = 15.0


class UtteranceGenerationClient:
    def __init__(self, hostports):
        self.hostports = hostports
        self.model_name = 'slug2slug'

        usr_dir.import_usr_dir('./transformer')
        problem = registry.problem('lang_gen')
        hparams = tf.contrib.training.HParams(data_dir='./transformer/t2t_data')
        problem.get_hparams(hparams)

        self.problem = problem

    def remove_unavailable_services(self):
        # Crate a dummy request
        request_empty = predict_pb2.PredictRequest()
        request_empty.model_spec.name = self.model_name

        # Keep only the services that are running
        hostports_running = []
        for hostport in self.hostports:
            if self.__verify_service(hostport, request_empty):
                hostports_running.append(hostport)

        self.hostports = hostports_running

        if len(self.hostports) == 0:
            print('Error: no running Slug2Slug service found.')
        else:
            print('----')
            print('Running Slug2Slug services:')
            print('\n'.join(self.hostports))
            print('----')
        sys.stdout.flush()

        return len(self.hostports)

    def get_utterance_for_mr(self, mr):
        '''Sends a request to the prediction services to produce an utterance.
        Args:
            mr: The query MR for which an utterance should be generated.
        Returns:
            The utterance generated from the query MR.
        '''

        print('Preprocessing the MR...', end=' ')
        sys.stdout.flush()
    
        # Delexicalize the input MR
        mr_tokens, mr_dict = tokenize_mr(mr)
        mr_delexed = ' '.join(mr_tokens)

        print('Done')
        print('Evaluating the query...', end=' ')
        sys.stdout.flush()

        # Generate candidate utterances for the input MR
        utt_candidates = self.__predict_single_input(mr_delexed, mr_dict)

        # If none of the services returned a valid response, terminate
        if len(utt_candidates) == 0:
            return None

        # DEBUG PRINT
        print(utt_candidates)

        # Find the utterance with the highest score among the candidates
        best_utt, best_score = max(utt_candidates, key=itemgetter(1))

        print('Done')
        print('Postprocessing the utterance...', end=' ')
        sys.stdout.flush()

        final_utt = finalize_utterance(best_utt, mr_dict)

        print('Done')

        return final_utt

    def __predict_single_input(self, mr, mr_dict):
        '''Encodes inputs, makes request to deployed TF model, and decodes outputs.'''

        # Retrieve the input encoder and output decoder
        input_encoder = self.problem.feature_info['inputs'].encoder
        output_decoder = self.problem.feature_info['targets'].encoder

        # Encode the input MR
        mr_encoded = self.__encode_input(mr, input_encoder)

        # Create a request for the prediction service
        request = self.__create_service_request(mr_encoded)

        if len(self.hostports) > 1:
            # Send the request to all running services in parallel
            pool = multiprocessing.Pool(len(self.hostports))
            results = pool.map_async(process_query, zip(self.hostports,
                                                        repeat(request),
                                                        repeat(output_decoder),
                                                        repeat(mr_dict)))
            pool.close()
            pool.join()

            # Gather the results from the services, and combine them into a single list
            utt_candidates = results.get()
            utt_candidates = list(itertools.chain(*utt_candidates))
        else:
            # Send the request to a single service
            utt_candidates = process_query((self.hostports[0],
                                            request,
                                            output_decoder,
                                            mr_dict))

        return utt_candidates

    def __verify_service(self, hostport, request_empty):
        stub = create_stub(hostport)

        try:
            stub.Predict(request_empty, RPC_TIMEOUT)
        except face.AbortionError as err:
            if err.code == interfaces.StatusCode.UNAVAILABLE:
                return False

        return True

    def __encode_input(self, mr, input_encoder):
        '''Encodes the input, and creates a TF Example record out of it.'''

        input_ids = input_encoder.encode(mr)
        input_ids.append(text_encoder.EOS_ID)

        features = {
            'inputs': tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids))
        }

        example = tf.train.Example(features=tf.train.Features(feature=features))

        return example.SerializeToString()

    def __create_service_request(self, mr_encoded):
        '''Assembles a request for the prediction service.'''

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto([mr_encoded], shape=[1]))

        return request


def process_query(task_args):
    '''Thread task for sending a request to the prediction service and processing the response.'''

    hostport, request, output_decoder, mr_dict = task_args

    # Define the connection to the prediction service
    stub = create_stub(hostport)

    try:
        # Send the request to the service
        response = stub.Predict(request, RPC_TIMEOUT)
    except:
        return []

    # Retrieve the beams
    predictions = tf.make_ndarray(response.outputs['outputs'])
    if predictions.ndim > 2:
        predictions = predictions.squeeze()

    # Truncate the beams from the EOS token (~ index 1 in the vocab) onwards
    predictions = [beam[:list(beam).index(1)] for beam in predictions]

    # Decode the beams into utterances
    utterances = [
        output_decoder.decode(beam) for beam in predictions
    ]

    # Retrieve the scores (log-probabilities) of the individual beams
    log_probs = tf.make_ndarray(response.outputs['scores'])
    if log_probs.ndim > 2:
        log_probs = log_probs.squeeze()

    # Update the utterance scores
    scores = [log_prob / scoreAlignment(utt, mr_dict) for utt, log_prob in zip(utterances, log_probs)]

    return list(zip(utterances, scores))


def create_stub(hostport):
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))

    return prediction_service_pb2.beta_create_PredictionService_stub(channel)


def main(_):
    if not FLAGS.query:
        print('Please, specify the query MR')
        return
    
    hostports = ['localhost:9000',
                 'localhost:9001',
                 'localhost:9002']

    client = UtteranceGenerationClient(hostports)

    # ---- SERVICE VALIDATION ----

    print('Verifying the services...')
    sys.stdout.flush()

    running_servers = client.remove_unavailable_services()
    if running_servers < 1:
        sys.exit()

    # ---- CLIENT QUERYING ----

    start_time = time.time()
    
    utterance = client.get_utterance_for_mr(FLAGS.query)
    print('****')
    print(utterance)
    print('****')

    print('[Running time:', (time.time() - start_time), 'seconds]')


if __name__ == '__main__':
    tf.app.run()
