# ANEA
The goal of Automatic (Named) Entity Annotation is to create a small annotated dataset for NER extracted from German domain-specific texts. 

To cite the related paper, please use 
```
@inproceedings{Zhukova2021a,
  title        = {ANEA: Automated (Named) Entity Annotation for German Domain-Specific Texts},
  author       = {Zhukova, Anastasia and Hamborg, Felix and Gipp, Bela},
  year         = 2021,
  month        = {September, 30th},
  booktitle    = {Proceedings of the 2nd Workshop on Extraction and Evaluation of Knowledge Entities from Scientific Documents (EEKE 2021) co-located with JCDL 2021, Virtual Event},
  publisher    = {CEUR},
  address      = {Illinois, USA},
  doi          = {10.6084/m9.figshare.17185373.v2},
  url          = {http://ceur-ws.org/Vol-3004/paper1.pdf},
  editor       = {Zhang, Chengzhi and Mayr, Philipp and Lu, Wei and Zhang, Yi}
}
```

## Installation and execution
Python 3.8 
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
