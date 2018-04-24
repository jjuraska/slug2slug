USR_DIR=./transformer
DATA_DIR=./data
TRAIN_DIR=./model

PROBLEM=lang_gen
MODEL=transformer
HPARAMS=transformer_lang_gen
BEAM_SIZE=4
ALPHA=0.6

t2t-exporter \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --problems=$PROBLEM \
    --data_dir=$DATA_DIR \
    --output_dir=$TRAIN_DIR \
    --t2t_usr_dir=$USR_DIR \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,write_beam_scores=True,return_beams=True"