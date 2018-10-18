## E2E NLG

The E2E NLG module leverages the seq2seq framework for end-to-end natural language generation from meaning representations (MRs). This is a work in progress...

---

##### USAGE

In the _e2e_nlg_ folder, optionally put your input files in the _data_ folder. Run _main.py_ in one of the following ways to run the training, or the evaluation (only after the training has been run):

```
python main.py --train [path_to_trainset] [path_to_devset]
python main.py --test [path_to_testset]
```

Replace [path_to_trainset], [path_to_devset], [path_to_testset] with relative paths to your trainset, devset, or testset, respectively. They are expected to be CSV files with two columns (their headers must be _mr_ and _ref_, respectively), the first containing the MRs, and the second containing the corresponding reference utterances.

Once the training is done, the _model_ folder will contain files describing the model, which will be used for evaluation. Therefore, you are not to modify them.

Finally, the evaluation produces output files in the _predictions_ folder. The _predictions.txt_ file contains raw results, while the _predictions_final.txt_ is produced during the postprocessing step.

---

###### REQUIREMENTS

- Python libraries: tensorflow, numpy, pandas, nltk, networkx
- NLTK modules: perluniprops, punkt
    - install using the following command: python -c "import nltk; nltk.download('[module_name]')"
