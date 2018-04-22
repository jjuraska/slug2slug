USR_DIR=./transformer
PROBLEM=lang_gen
DATA_DIR=./transformer/t2t_data
TMP_DIR=./transformer/t2t_tmp
mkdir -p $DATA_DIR $TMP_DIR

t2t-datagen \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM
