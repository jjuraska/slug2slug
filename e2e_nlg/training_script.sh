python ../bin/train.py \
  --config_paths="
      ./config.yml" \
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
  --output_dir ./model