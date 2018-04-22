PROBLEM=lang_gen
MODEL=transformer
HPARAMS=transformer_lang_gen

USR_DIR=./transformer
DATA_DIR=./transformer/t2t_data
TRAIN_DIR=./transformer/t2t_train/$PROBLEM-$HPARAMS

t2t-exporter \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --problems=$PROBLEM \
    --data_dir=$DATA_DIR \
    --output_dir=$TRAIN_DIR \
    --t2t_usr_dir=$USR_DIR \
    --decode_hparams="beam_size=10,alpha=0.6,write_beam_scores=True,return_beams=True"