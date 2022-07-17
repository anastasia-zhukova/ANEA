# Gradual fine-tuning a deep NER model on domain specific entities

This project aims to train a classifier model to indentify domain specific entities in text. It is based on the output from ANEA [[1]](#acknowledgements), meaning that the categories that ANEA outputs in the final_categories.json will be used as domain specific entity tags. Since domain specific data is a scarce resource, general data and domain specific data are mixed for training, in a gradual fine-tuning approach as shown in [[2]](#acknowledgements)

- "_MEDIC approach_": The classifier model used in this project, based on the idea from Nayel et. al. [[3]](#acknowledgements)

## Requirements

This installation is tested on a python 3.8 environment, in this case using [Anaconda](https://www.anaconda.com) but it should also work with just python 3.8 and pip.

First install basic requirements:
```install
pip install numpy pandas seaborn matplotlib tqdm ftfy sklearn datasets gdown gensim
```
Then decide whether you can use CUDA support or not.

Pytorch and SpaCy without CUDA support:
```install
pip3 install torch torchvision torchaudio
pip install torchcrf
pip install -U pip setuptools wheel
pip install -U spacy
```

Pytorch and SpaCy with CUDA support:
```install
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchcrf
pip install -U pip setuptools wheel
pip install -U 'spacy[cuda116]'
```

Download SpaCy model:
``` install
python -m spacy download de_core_news_sm
```



## Training & Evaluation

To train and evaluate the models you can use the experiment.py as follows

```train
python run_model.py
```
This will start the program and ask you to enter a topic and date from a previous ANEA execution, which will then be used as entity classes and in-domain data to train and evaluate the model. The results from this experiment will be saved to your disk.
Currently training and evaluation will always be done in one experiement because models can not be saved.

### Options
In the current development state a few more options are available. Note that for some arguments it is possible to give multiple options, this will result in a very simple gridsearch where all different combinations for the given parameters will be tried out iteratively (may result in very long runtimes!). For a more detailed overview over the command line options, please use ```run_module -h``` or ```run_module --help```

To select a topic to use (if this is set then you will not be prompted by the program):
```options
-t | --topic [string]
```
Set the date of an ANEA execution which will be used:
```options
-d | --date [string, 'YYYY-mm-dd']
```
Batchsize for training and evaluation:
```options
--batchsize [any integer]
```
Number of traning epochs:
```options
--epochs [any integer]
```
Use a random seed to select the same data when running multiple experiments:
```options
--seed [any integer]
```
when omitted it will be completely random.

Relative size of the training and test set size (sum of both should not be > 1):
```options
--trainsize [any float between 0 and 1]
```
and
```options
--testsize [any float between 0 and 1]
```
How many models will be initialized and trained for each combination of hyperparameters:
```options
--trials [any integer > 0]
```
Select average for evaluation metrics:
```options
--avg ['macro'|'none'|'binary'|'weighted'|'samples']
```
Learning rate:
```options
-lr | --learning_rate [any float]
```
Period of learning rate decay in epochs:
```options
--step_size [any int]
```
To use the label distribution as weights for the loss funciton add this option:
```options
--label_dist
```
Number of fine tuning steps with general data, this will determine how much general data is mixed with domain data each step. Also one more fine-tuning step will always be done, where the model is fine tuned only on domain data:
```options
--fine_tuning_steps [any int]
```
To save the results as .json instead of .csv:
```options
--json
```
Choose one or many of the prepared datasetes from HuggingFace as general data:
```options
--hf_datasets ['News'|'Amazon'|'Books']
```
This will use either [News](https://huggingface.co/datasets/news_commentary/viewer/de-es/train), [Amazon](https://huggingface.co/datasets/amazon_reviews_multi/viewer/de/train) and/or [Books](https://huggingface.co/datasets/opus_books/viewer/de-en/train).
You can use hf_limit to limit the maximum amount of samples used from the HF datasets:
```options
--hf_limit [any integer]
```
Alternatively you can use (almost) any dataset from HuggingFace via:
```options
--own_hf_dataset [path] [name] [split] [colname_data] [lang] [limit] [shuffle]
```
And it is also possible to use different texts from the ANEA output as additional general data. Just run ANEA for different topics and give those topics as input:
```options
--anea_datasets [str, any topic]
```

## Results
### Program output
```run_model``` will save a csv file with all the training and evaluation results and the used hyperparameters. In the folder of the ANEA execution results, under the date and topic that were used will be a new folder _ML_classifier_results_ that contains the csv file(s).

### Plot results
To plot the results of multiple runs the ```plot_results``` module may be used. One or multiple result files that used the same (ANEA) date and topic can be given as input:
```usage
plot_results --path \PATH\TO\RESULTS\results1 \PATH\TO\RESULTS\results2 ...
```
Example:
```exmaple
python -m plot_results evaluation\execution_results\2022-05-22\Datenbanken\ML_classifier_results\2022-07-11_13-05-42_results_News_15k.csv evaluation\execution_results\2022-05-22\Datenbanken\ML_classifier_results\2022-07-11_19-16-56_results_News_5k.csv evaluation\execution_results\2022-05-22\Datenbanken\ML_classifier_results\2022-07-11_19-33-59_results_News_1k.csv
```

## Acknowledgements
[1] A. Zhukova, „ANEA: Automated (Named) Entity Annotation for German Domain-Specific Texts“, S. 10, 2021.

[2] H. A. Nayel und S. H. L, „Integrating Dictionary Feature into A Deep Learning Model for Disease Named Entity Recognition“, arXiv:1911.01600 [cs], Nov. 2019, Zugegriffen: 30. Oktober 2021. [Online]. Verfügbar unter: http://arxiv.org/abs/1911.01600

[3] H. Xu, S. Ebner, M. Yarmohammadi, A. S. White, B. V. Durme, und K. Murray, „Gradual Fine-Tuning for Low-Resource Domain Adaptation“, S. 8.





## Contributing

The project is published under the [Apache License 2.0](EfficientLearning/LICENSE-2.0.txt).
