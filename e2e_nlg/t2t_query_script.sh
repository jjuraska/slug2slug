t2t-query-server \
  --server=localhost:9001 \
  --servable_name=slug2slug \
  --problem=lang_gen \
  --data_dir=./transformer/t2t_data \
  --t2t_usr_dir=./transformer \
  --timeout_secs=30 \
  --inputs_once="name &slot_vow_name& area city centre familyfriendly no"
  