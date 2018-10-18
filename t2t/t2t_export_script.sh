USR_DIR=./t2t
DATA_DIR=./data
TRAIN_DIR=./model

PROBLEM=lang_gen
#PROBLEM=lang_gen_multi_vocab

MODEL=transformer
#MODEL=lstm_seq2seq_attention_bidirectional_encoder

HPARAMS=transformer_lang_gen
#HPARAMS=transformer_lang_gen_multi_vocab
#HPARAMS=lstm_lang_gen

BEAM_SIZE=4
ALPHA=1.0

t2t-exporter \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --output_dir=$TRAIN_DIR \
    --t2t_usr_dir=$USR_DIR \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,write_beam_scores=False,return_beams=False"
