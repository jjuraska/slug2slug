python ../bin/infer.py \
  --tasks "
    - class: DecodeText" \
  --model_dir ./model \
  --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - ./data/test_source.txt" \
  >  ./predictions/predictions.txt