USR_DIR=./t2t
DATA_DIR=./data
TRAIN_DIR=./model
PRETRAINED_DIR=$TRAIN_DIR/_pretrained

PROBLEM=lang_gen
#PROBLEM=lang_gen_multi_vocab

MODEL=transformer
#MODEL=lstm_seq2seq_attention_bidirectional_encoder

HPARAMS=transformer_lang_gen
#HPARAMS=transformer_lang_gen_finetune
#HPARAMS=transformer_lang_gen_multi_vocab
#HPARAMS=lstm_lang_gen

mkdir -p $TRAIN_DIR

t2t-trainer \
    --data_dir=$DATA_DIR \
    --t2t_usr_dir=$USR_DIR \
    --output_dir=$TRAIN_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --train_steps=10000 \
    --local_eval_frequency=200 \
    --eval_throttle_seconds=1 \
    --eval_steps=100 \
    --eval_early_stopping_metric=accuracy \
    --eval_early_stopping_metric_delta=0.1 \
    --eval_early_stopping_steps=3000 \
    --eval_early_stopping_metric_minimize=False \
    --keep_checkpoint_max=100
#    --iterations_per_loop=100
#    --warm_start_from=$PRETRAINED_DIR
