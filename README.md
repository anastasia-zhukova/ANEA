# ANEA
The goal of Automatic (Named) Entity Annotation is to create a small annotated dataset for NER extracted from German domain-specific texts. 

## Installation and execution
Python 3.7 
Required approx. 8Gb of hard memory, 16Gb RAM

Download "numberbatch_voc.txt" from  https://drive.google.com/file/d/1Ag3gQUBtmqB-WAGXk67nJwUvMiZ1DdQG/view?usp=sharing
and place to
```
resources/numberbatch
```

You can either use your own documents stored as a list of strings in a json file, or use a key-word for searching in Wikipedia to get articles to annotate. 
Place your file into ```data``` folder.

Then execute
```
pip install -r requirements.txt
```
```
python -m spacy download de_core_news_sm
```
```
run_anea.py
```
Follow the instructions to choose a folder with your topic to annotate.
