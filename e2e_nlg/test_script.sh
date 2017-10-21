python ../bin/infer.py \
  --tasks "
    - class: DecodeText " \
  --model_dir ./model \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - ./data/test_source.txt" \
#  >  ./predictions/predictions.txt