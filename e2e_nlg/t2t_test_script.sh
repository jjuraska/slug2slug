USR_DIR=./transformer
DATA_DIR=./data
TRAIN_DIR=./model

DECODE_FILE=$DATA_DIR/test_source.txt
PREDICTION_FILE=./predictions/predictions.txt

PROBLEM=lang_gen
MODEL=transformer
HPARAMS=transformer_lang_gen
BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --t2t_usr_dir=$USR_DIR \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
    --decode_from_file=$DECODE_FILE \
    --decode_to_file=$PREDICTION_FILE
