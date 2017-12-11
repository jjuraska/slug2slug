python ../bin/train.py \
  --config_paths="
      ./config_rnn.yml,
      ./config_metrics.yml" \
  --model_params "
      vocab_source: ./data/vocab_source.txt
      vocab_target: ./data/vocab_target.txt" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - ./data/training_source.txt
      target_files:
        - ./data/training_target.txt" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - ./data/dev_source.txt
       target_files:
        - ./data/dev_target.txt" \
  --batch_size 64 \
  --train_steps 20000 \
  --output_dir ./model \
  --eval_every_n_steps 2000 \
  --save_checkpoints_steps 1000 \
  --keep_checkpoint_max 10 \
  --keep_checkpoint_every_n_hours 2