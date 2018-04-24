USR_DIR=./transformer
DATA_DIR=./data
TMP_DIR=./transformer/t2t_tmp

PROBLEM=lang_gen

mkdir -p $DATA_DIR $TMP_DIR

t2t-datagen \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM
