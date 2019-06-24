USR_DIR=./t2t
DATA_DIR=./data
TRAIN_DIR=./model

DECODE_FILE=$DATA_DIR/test_source.txt
PREDICTION_FILE=./predictions/predictions.txt

PROBLEM=lang_gen
#PROBLEM=lang_gen_multi_vocab

MODEL=transformer
#MODEL=lstm_seq2seq_attention_bidirectional_encoder

HPARAMS=transformer_lang_gen
#HPARAMS=transformer_lang_gen_finetune
#HPARAMS=transformer_lang_gen_multi_vocab
#HPARAMS=lstm_lang_gen

BEAM_SIZE=10
ALPHA=1.0

t2t-decoder \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --t2t_usr_dir=$USR_DIR \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,write_beam_scores=False,return_beams=False" \
    --decode_from_file=$DECODE_FILE \
    --decode_to_file=$PREDICTION_FILE
