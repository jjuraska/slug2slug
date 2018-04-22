PROBLEM=lang_gen
MODEL=transformer
HPARAMS=transformer_lang_gen

USR_DIR=./transformer
DATA_DIR=./transformer/t2t_data
TRAIN_DIR=./transformer/t2t_train/$PROBLEM-$HPARAMS
mkdir -p $TRAIN_DIR

t2t-trainer \
    --data_dir=$DATA_DIR \
    --t2t_usr_dir=$USR_DIR \
    --output_dir=$TRAIN_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --train_steps=11000 \
    --eval_steps=100 \
    --local_eval_frequency=500 \
    --eval_early_stopping_metric=loss \
    --eval_early_stopping_metric_delta=0.05 \
    --eval_early_stopping_steps=3 \
    --keep_checkpoint_max=10
