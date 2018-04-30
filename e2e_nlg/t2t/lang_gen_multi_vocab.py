import os
import io
import tensorflow as tf

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import config


@registry.register_hparams
def transformer_lang_gen_multi_vocab():
    hparams = transformer.transformer_base()

    hparams.num_hidden_layers = 2
    hparams.hidden_size = 256
    hparams.filter_size = 512
    hparams.num_heads = 8
    hparams.attention_dropout = 0.6
    hparams.layer_prepostprocess_dropout = 0.6
    hparams.learning_rate = 0.05
    # hparams.batch_size = 64             # default: 4096
    hparams.max_input_seq_length = 50
    hparams.max_target_seq_length = 60
    # hparams.min_length_bucket = 10      # default: 0
    hparams.shared_embedding_and_softmax_weights = False

    return hparams


@registry.register_problem
class LangGenMultiVocab(text_problems.Text2TextProblem):
    """Generate a natural language utterance from a structured meaning representation (MR)."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def oov_token(self):
        return 'UNK'

    @property
    def vocab_input_filename(self):
        return 'vocab.lang_gen_multi_vocab.source'

    @property
    def vocab_target_filename(self):
        return 'vocab.lang_gen_multi_vocab.target'

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.vocab_input_filename)
        target_vocab_filename = os.path.join(data_dir, self.vocab_target_filename)

        source_token = text_encoder.TokenTextEncoder(source_vocab_filename)
        target_token = text_encoder.TokenTextEncoder(target_vocab_filename)

        return {
            'inputs': source_token,
            'targets': target_token,
        }

    @property
    def is_generate_per_split(self):
        # If False, generate_data will shard the data into TRAIN and EVAL for us
        return True

    @property
    def dataset_splits(self):
        return [{
            'split': problem.DatasetSplit.TRAIN,
            'shards': 10,
        }, {
            'split': problem.DatasetSplit.EVAL,
            'shards': 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        training_source_file = os.path.join(config.DATA_DIR, 'training_source.txt')
        training_target_file = os.path.join(config.DATA_DIR, 'training_target.txt')
        dev_source_file = os.path.join(config.DATA_DIR, 'dev_source.txt')
        dev_target_file = os.path.join(config.DATA_DIR, 'dev_target.txt')

        train = dataset_split == problem.DatasetSplit.TRAIN
        source_file = (training_source_file if train else dev_source_file)
        target_file = (training_target_file if train else dev_target_file)

        # def generator_samples_content(get_source, get_target):
        #     source, target = None, None
        #
        #     with tf.gfile.GFile(source_file, mode='r') as f_x_train, \
        #             tf.gfile.GFile(target_file, mode='r') as f_y_train:
        #
        #         mrs = f_x_train.read().splitlines()
        #         utterances = f_y_train.read().splitlines()
        #
        #         for mr, utt in zip(mrs, utterances):
        #             yield mr, utt
        #
        # def generator_source():
        #     for source, _ in generator_samples_content(False, True):
        #         yield source.strip()
        #
        # def generator_target():
        #     for _, target in generator_samples_content(False, True):
        #         yield target.strip()
        #
        # # Generate vocab for both source and target
        # source_vocab = generator_utils.get_or_generate_vocab_inner(
        #     data_dir=data_dir,
        #     vocab_filename=self.vocab_input_filename,
        #     vocab_size=self.input_vocab_size,
        #     generator=generator_source())
        #
        # target_vocab = generator_utils.get_or_generate_vocab_inner(
        #     data_dir=data_dir,
        #     vocab_filename=self.vocab_target_filename,
        #     vocab_size=self.target_vocab_size,
        #     generator=generator_target())
        #
        # with io.open('data/training_source.txt', 'r', encoding='utf8') as f_x_train, \
        #         io.open('data/training_target.txt', 'r', encoding='utf8') as f_y_train:
        #     mrs = f_x_train.read().splitlines()
        #     utterances = f_y_train.read().splitlines()
        #
        #     for mr, utt in zip(mrs, utterances):
        #         yield {
        #             'inputs': mr,
        #             'targets': utt
        #         }

        return text_problems.text2text_txt_iterator(source_file, target_file)

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        source_encoder = self.feature_encoders(data_dir)['inputs']
        target_encoder = self.feature_encoders(data_dir)['targets']

        return text_problems.text2text_generate_encoded(generator,
                                                        source_encoder,
                                                        targets_vocab=target_encoder,
                                                        has_inputs=True)
