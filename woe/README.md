# IR22 PA2

In this directory of the repository you will find 2 different approaches:
1. Transformers
2. Clustering

## 1. Transformers
The directory 'transformer' contains all necessary code to train a transformer model on the ANEA data. 
The script `train_ner_model.py` trains a model, but needs the package `simpletransformers` installed. But after Ms. Zhukova said I should try to run a transformer with HuggingFace, it was not kept up-to-date with the changes in the preprocessing of the data! So you might run into problems running this script.

The script `train_transformer_with_huggingface.py` creates a HuggingFace dataset & trains a transformer with a HuggingFace Trainer. <br>
In line 47 `used_model = "distilbert-base-german-cased"` you can set the used base Transformer model which you want to train for your NER model.

After this the settings for the trainer and the TrainingArguments are set as followed:
```
args = TrainingArguments(
        "MyOwnTransformerModel",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to=None,
        overwrite_output_dir=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
```


## 2. Clustering

The directory 'clustering' contains the script to try different clustering approaches on the data.
If you plan to use the fastText model to get word vectors, please download the model [here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz).
Place this model in the directory `woe/fasttext` (or adapt paths accordingly).
Alternatively you can always use another model from [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) & adapt the script, as described in a later part.
To start the algorithm just run the script `with_clustering.py`. 

### Changing the clustering parameters
In order to change the parameters of the clustering algorithm you want to run, please take a look into `KEYWORD_ARGUMENTS.py`.
In this file, you will see a dictionary with the three clustering algorithms as keys. For each key, the corresponding value is a list of dicts. You can add (or remove) new dicts with the parameters to this list.
Its important that you name the keys in the dictionary 100% the same as the parameters in the clustering mechanism are.

If you are unsure which keyword arguments you are allowed to use, please take a look [here](https://scikit-learn.org/stable/modules/classes.html#classes).


### Adding a clustering algorithm
In order to add a new clustering algorithm, please change line 159 in the `with_clustering.py` file.
`for algorithm in ['kMeans', 'hierarchical', 'affinity']` - add a new, unique string into this array, of course its recommended to use the name of the algorithm. <br>
You will also need add the same new, unique string you added to the array, to the dictionary in `KEYWORD_ARGUMENTS.py`, which will return an empty list. <br>
Starting in line 169 you will find an if-else, where you will also need the newly added clustering algorithm to the variable `algo`, `if algorithm == new, unique string`

E.g.: You want to add the clustering algorithm `SpectralClustering`:
1. Change the array: `for algorithm in ['kMeans', 'hierarchical', 'affinity', 'spectral']`
2. Add the name to the dictionary: `KEYWORD_ARGUMENTS = { 'spectral': [], 'kMeans': ... }`
3. Change the if else:
```
if algorithm == 'kMeans':
   algo = KMeans(n_clusters=len(orig_labels), **kwargs)
elif algorithm == 'hierarchical':
   algo = AgglomerativeClustering(n_clusters=len(orig_labels), **kwargs)
elif algorithm == 'spectral':
   algo = SpectralClustering(**kwargs)
else:
   algo = AffinityPropagation(**kwargs)
```

### Adding a new embedding
If you want to implement a new form of embedding you need to adapt the `Embedding_Wrapper.py`.
Add a new if statement to the `__init__` function of the wrapper, where you assign the newly chosen embedder to self.embedder `if mode == newly, chosen keyword for the embedder`.  
In the `get_word_vector(self, word)` function you need to add some lines, where you specify what exactly needs to be done to return the embeddings as you want to use them.
Last, but not least, you will need to add the newly chosen keyword to the array in line 105.
`for embed in [EMBEDDING KEYWORDS]`

E.g.: You want to add a Word2Vec model from gensim:
1. Add the if statement to the init function of the wrapper:
```
if mode == 'gensim':
   self.embedder = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100, window = 5)
```
2. Adapt the `get_word_vector(self, word)` function:
```
if self.mode == 'gensim':
    return self.embedder.wv[word]
```
3. Adapt the array in line 105: `for embed in ['fasttext', 'nlp', 'gensim']:`

(Please be advised that the gensim model has not been tested, so you might need to change what the `get_word_vector` returns in order to implement the gensim model correctly)


