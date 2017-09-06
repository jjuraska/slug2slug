python ../bin/infer.py \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: predictions/beams.npz" \
  --model_dir ./model \
  --model_params "
    inference.beam_search.beam_width: 10" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - ./data/test_source.txt" \
  >  ./predictions/predictions.txt