USR_DIR=./t2t
DATA_DIR=./data
TMP_DIR=./t2t/tmp

PROBLEM=lang_gen
#PROBLEM=lang_gen_multi_vocab

t2t-datagen \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM
