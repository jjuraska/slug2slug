# Slug2Slug

Slug2Slug is an end-to-end system for data-to-text natural language generation (NLG) from structured meaning representations (MRs). It builds on top of a neural encoder-decoder model, and uses adaptive delexicalization, semantic reranking, and other techniques, to achieve higher syntactic correctness and semantic accuracy of generated utterances.

Slug2Slug is the overall winner of the 2017 [E2E NLG Challenge](http://www.macs.hw.ac.uk/InteractionLab/E2E/).

## Usage

### Requirements

- Python 3.6 or above
- Python libraries: tensor2tensor, tensorflow, numpy, pandas, nltk, networkx
- NLTK modules: perluniprops, punkt
    - Install using the following command: `python -c "import nltk; nltk.download('[module_name]')"`, where you replace `[module_name]` with the corresponding module name.

### Configuration

Model configuration files, as well as training and inference configuration files, can be found in the _t2t_ directory.

### Training & Inference

Run _run_task.py_ in one of the following ways to start training or evaluating a model, respectively:

```
python run_task.py --train [path_to_trainset] [path_to_devset]
python run_task.py --test [path_to_testset]
```

Replace `[path_to_trainset]`, `[path_to_devset]`, `[path_to_testset]` with relative paths to your training set, validation set, or test set, respectively. They are expected to be CSV files with two columns (their headers must be _mr_ and _ref_, respectively), the first containing the MRs, and the second containing the corresponding reference utterances.

Once the training is done, the _model_ directory will contain the trained model files, which can then be used for inference. Running inference produces output files in the _predictions_ directory. The _predictions.txt_ file contains raw results, while the _predictions_final.txt_ is produced during the postprocessing step.

## Notice

- Slug2Slug is highly experimental and in continuous development. If you find a bug, feel free to contact me (see email address in the paper below), or open an issue.
- Since the submission to the E2E NLG Challenge we have switched from using the seq2seq framework to tensor2tensor, and with it, from an LSTM-based model to a Transformer-based model. Moreover, our slot aligner has been revamped. If you are looking for the version of the system submitted to the challenge, you can try using the commit from Dec 15, 2017. However, keep in mind that the usage instructions above will not apply to that version.

## Citing Slug2Slug

If you use or refer to the Slug2Slug system or any of its parts, please cite [this paper](https://www.aclweb.org/anthology/N18-1014/):

- Juraj Juraska, Panagiotis Karagiannis, Kevin Bowden, and Marilyn Walker. 2018. A Deep Ensemble Model with Slot Alignment for Sequence-to-Sequence Natural Language Generation. In _Proceedings of the 2018 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_.

## License

Licensed under the MIT License (see [LICENSE](LICENSE)).
