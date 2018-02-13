from __future__ import print_function

import sys
import multiprocessing
import time
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
from grpc.beta import implementations, interfaces
from grpc.framework.interfaces.face import face
from itertools import repeat

from data_loader import tokenize_mr
from slot_alignment import scoreAlignment
from postprocessing import finalize_utterance


tf.app.flags.DEFINE_string('query', '', 'Input MR')
FLAGS = tf.app.flags.FLAGS

RPC_TIMEOUT = 5.0


class UtteranceGenerationClient:
    def __init__(self, hostports):
        self.model_name = 'slug2slug'
        self.signature_name = 'predict_utterance'
        self.hostports = hostports


    def get_utterance_from_mr(self, mr):
        '''
        Sends a request to the prediction services to produce an utterance.
        Args:
            mr: THe query MR for which an utterance should be generated.
        Returns:
            The utterance generated from the query MR.
        '''

        print('Preprocessing the MR...', end=' ')
        sys.stdout.flush()
    
        # convert the MR into a sequence of tokens
        mr_tokens, mr_dict = tokenize_mr(mr)

        print('Done')
        print('Evaluating the query...', end=' ')
        sys.stdout.flush()
    
        # construct the request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name

        request.inputs['mr_tokens'].CopyFrom(tf.contrib.util.make_tensor_proto(mr_tokens, shape=[1, len(mr_tokens)]))
        request.inputs['mr_len'].CopyFrom(tf.contrib.util.make_tensor_proto(len(mr_tokens), shape=[1]))
    
        # send the request to all running services in parallel
        pool = multiprocessing.Pool(len(self.hostports))
        results = pool.map_async(process_query, zip(self.hostports, repeat(request), repeat(mr_dict)))
        pool.close()
        pool.join()

        print('Done')

        # gather the results from the services
        utt_candidates = results.get()

        # DEBUG PRINT
        #print(utt_candidates)

        # if none of the services returned a valid response, return None
        if all(utt[0] is None for utt in utt_candidates):
            return None

        print('Re-ranking candidate utterances...', end=' ')
        sys.stdout.flush()

        # re-rank the candidate utterances
        best_utt = ''
        best_score = -1
        for utt, utt_score in utt_candidates:
            if utt_score > best_score:
                best_utt, best_score = utt, utt_score
    
        print('Done')
        print('Postprocessing the utterance...', end=' ')
        sys.stdout.flush()

        final_utt = finalize_utterance(best_utt, mr_dict)

        print('Done')

        return final_utt


    def remove_unavailable_services(self):
        # crate a dummy request
        request_empty = predict_pb2.PredictRequest()
        request_empty.model_spec.name = self.model_name
        
        # verify if all services are running
        hostports_running = []
        for hostport in self.hostports:
            if self.__verify_service(hostport, request_empty):
                hostports_running.append(hostport)

        if len(hostports_running) == 0:
            print('Error: no running service found.')
            sys.exit()

        self.hostports = hostports_running
    
        print('----')
        print('Running services:')
        print('\n'.join(self.hostports))
        print('----')
        sys.stdout.flush()


    def __verify_service(self, hostport, request_empty):
        host, port = hostport.split(':')
        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        try:
            stub.Predict(request_empty, RPC_TIMEOUT)
        except face.AbortionError as err:
            if err.code == interfaces.StatusCode.UNAVAILABLE:
                return False

        return True


def process_query(task_args):
    hostport, request, mr_dict = task_args

    # define the connection to the service
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    try:
        response = stub.Predict(request, RPC_TIMEOUT)
    except:
        return None, 0

    # retrieve the list of tokens representing the utterance
    utt_tokens = response.outputs['utterance'].string_val
    # remove the sequence-end token and convert to a string

    utt_temp = []
    for t in utt_tokens:
        if type(t) is str:
            break
        utt_temp.append(t.decode())
    utt_tokens = utt_temp
    utt = ' '.join(utt_tokens[:-1]).strip()
    # score the utterance
    utt_score = scoreAlignment(utt, mr_dict)

    return utt, utt_score


def main(_):
    if not FLAGS.query:
        print('Please, specify the query MR')
        return
    
    hostports = ['localhost:9000',
                 'localhost:9001',
                 'localhost:9002']
    

    client = UtteranceGenerationClient(hostports)

    # ---- SERVICE VALIDATION ----

    #print('Verifying the services...')
    #sys.stdout.flush()

    #client.remove_unavailable_services()


    # ---- CLIENT QUERYING ----

    start_time = time.time()
    
    utterance = client.get_utterance_from_mr(FLAGS.query)
    print('****')
    print(utterance)
    print('****')

    print('[Running time:', (time.time() - start_time), 'seconds]')


if __name__ == '__main__':
    tf.app.run()
