USR_DIR=./t2t
DATA_DIR=./data

PROBLEM=lang_gen
SERVABLE_NAME=slug2slug

t2t-query-server \
  --server=localhost:9001 \
  --servable_name=$SERVABLE_NAME \
  --problem=$PROBLEM \
  --data_dir=$DATA_DIR \
  --t2t_usr_dir=$USR_DIR \
  --timeout_secs=15 \
  --inputs_once="name &slot_vow_name& area city centre familyfriendly no"
  