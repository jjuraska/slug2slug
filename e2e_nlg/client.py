"""A client that talks to tensorflow_model_server loaded with mnist model.
The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.
Typical usage example:
        mnist_client.py --server=localhost:9000
"""

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/home/juriq/Git/seq2seq/e2e_nlg/saved_model', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


class _Result(object):
    """Counter for the prediction results."""

    def __init__(self, concurrency):
        self._utterance = ''
        self._concurrency = concurrency
        self._condition = threading.Condition()

    def set_utterance(self, val):
        self._utterance = val

    def get_utterance(self):
        return self._utterance

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result):
    """Creates RPC callback function.
    Args:
        label: The correct label for the predicted example.
    Returns:
        The callback function.
    """
    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
            result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = result_future.result().outputs[0]
            result.set_utterance(response)

    return _callback


def do_inference(hostport, work_dir, concurrency):
    """Tests PredictionService with concurrent requests.
    Args:
        hostport: Host:port address of the PredictionService.
        work_dir: The full path of working directory for test data set.
        concurrency: Maximum number of concurrent requests.
    Returns:
        The classification error rate.
    Raises:
        IOError: An error occurred processing test data set.
    """

    #test_data_set = mnist_input_data.read_data_sets(work_dir).test
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result = _Result(concurrency)
    
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'slug2slug'
    #request.model_spec.signature_name = 'predict_images'
    
    mr = 'name &slot_vow_name& area city centre familyfriendly no'
    label = 'There is a place in the city centre, Alimentum, that is not family-friendly.'

    #print(request.inputs.keys()[0])
    request.inputs[0].CopyFrom(tf.contrib.util.make_tensor_proto(mr, shape=[1, 1]))
    #result.throttle()
    result_future = stub.Predict.future(request, 5.0)    # 5 seconds
    result_future.add_done_callback(_create_rpc_callback(label))

    return result.get_utterance(), label


def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return

    utterance, label = do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency)
    print('\nGenerated utterance:', utterance)
    print('\nReference utterance:', label)


if __name__ == '__main__':
    tf.app.run()
