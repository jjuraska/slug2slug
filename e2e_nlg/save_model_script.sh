python ../bin/save_model.py \
  --tasks "
    - class: DecodeText" \
  --model_dir ./model \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - ./data/test_source.txt"