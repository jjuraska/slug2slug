USR_DIR=./t2t
DATA_DIR=./data
TRAIN_DIR=./model
PREDICTIONS_DIR=./predictions/batch

DECODE_FILE=$DATA_DIR/test_source.txt

PROBLEM=lang_gen
#PROBLEM=lang_gen_multi_vocab

MODEL=transformer
#MODEL=lstm_seq2seq_attention_bidirectional_encoder

HPARAMS=transformer_lang_gen
#HPARAMS=transformer_lang_gen_multi_vocab
#HPARAMS=lstm_lang_gen

BEAM_SIZE=4
ALPHA=1.0

t2t-translate-all \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --model_dir=$TRAIN_DIR \
    --t2t_usr_dir=$USR_DIR \
    --beam_size=$BEAM_SIZE \
    --alpha=$ALPHA \
    --write_beam_scores=False \
    --return_beams=False \
    --source=$DECODE_FILE \
    --translations_dir=$PREDICTIONS_DIR \
#    --decoder_command="python C:/Users/JuriQ/.virtualenvs/slug2slug/Scripts/t2t-decoder {params}"       # Might require configuring, depending on your system
