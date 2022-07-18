
"""
1. Get new texts regarding the selected topic (see below)
2. Extract noun phrases
3. Vectorize them
  (e.g., with fasttext or take BERT vector of each token of interesting within its context)
4. Cluster them into the existing clusters from ANEA (try different clustering approaches,
  e.g., kNN to find to which category/cluster each term belong best)
5. Suggest a user to have a look at the results in the visualization tool.
"""
import json
import os
import pickle
import re
import string

import ftfy
import numpy as np
import pandas as pd
import spacy
from HanTa import HanoverTagger as ht
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import f1_score, homogeneity_completeness_v_measure
from tqdm import tqdm
from sklearn import preprocessing

from Embedding_Wrapper import Embedding_Wrapper
from KEYWORD_ARGUMENTS import KEYWORD_ARGUMENTS

nlp = spacy.load('de_core_news_sm')


def _filter_terms(annot_text):
    term_array = []
    with tqdm(total=len(annot_text)) as pbar:
        for doc in annot_text:
            for t in doc:
                if t.pos_ not in ['NOUN']:
                    if t.pos_ != "VERB":
                        continue
                    else:
                        if t.orth_[-2:] != "er" and t.orth_[-3:] != "ung":
                            continue

                if len(t.orth_) < 4 or len(set(t.orth_)) < 2:
                    # ignore too short words and test cases like "aaaaa"
                    continue

                if len(re.findall(r'[a-zA-Z]+\d+', t.orth_)):
                    # ignore funcloc codes like A123
                    continue

                if len(re.findall(r'\d', t.orth_)):
                    # ignore if there is a digit
                    continue

                if len(re.findall(r'#|:', t.orth_)):
                    # ignore if there is "#" or ":"
                    continue

                word = t.orth_[1:].capitalize() if t.orth_[0] in string.punctuation else t.orth_
                word = word.split(";")[-1].capitalize() if len(word.split(";")) > 1 else word
                tags = ht.HanoverTagger('morphmodel_ger.pgz').tag_sent([word])
                lemma = tags[0][1]

                if lemma not in term_array:
                    term_array.append(lemma)
            pbar.update()
    return term_array


def preprocess(new):
    known_dir = os.path.join("..", "data", "already_known")
    unknown_dir = os.path.join("..", "data", "unknown")
    if not new:
        text_file = os.path.join(known_dir, "databases", "databases_texts.json")
        label_file = os.path.join(known_dir, "databases", "databases_labels.json")
    else:
        text_file = os.path.join(unknown_dir, "Datenbanken.json")
        label_file = os.path.join(unknown_dir, "final_categories.json")

    with open(text_file, "r", encoding='utf8') as f:
        orig_texts = json.load(f)

    annot_texts = []
    for text in orig_texts:
        annot_texts.append(nlp(ftfy.fix_text(text)))

    with open(label_file, "r", encoding='utf8') as f:
        labels = json.load(f)

    terms = _filter_terms(annot_texts)
    return terms, labels


if __name__ == '__main__':
    already_known_terms, orig_labels = preprocess(False)
    unknown_terms, unknown_labels = preprocess(True)
    # with open("tmp.pkl", "rb") as f:
    #     (already_known_terms, orig_labels) = pickle.load(f)
    #
    # with open("u_tmp.pkl", "rb") as f:
    #     (unknown_terms, unknown_labels) = pickle.load(f)

    for embed in ['fasttext', 'nlp']:
        print(f"Embeddings from: {embed}")
        embedder = Embedding_Wrapper(embed)

        """ known terms """
        vectorized_terms = []
        categories = []
        tmp_terms = []
        for term in already_known_terms:
            to_append_as_category = 'O'
            for key, value in orig_labels.items():
                if term in value:
                    to_append_as_category = key
                    break

            if to_append_as_category != 'O':
                categories.append(to_append_as_category)
                vectorized_term = embedder.get_word_vector(term)
                tmp_terms.append(term)
                vectorized_terms.append(vectorized_term)

        vectorized_terms = preprocessing.normalize(vectorized_terms)
        ar = []
        for npar in vectorized_terms:
            ar.append(npar)

        """ unknown terms """
        vec_unknown_terms = []
        tmp_unknown_terms = []
        unknown_categories = []
        for u_term in unknown_terms:
            to_append_as_category = 'O'
            for key, value in orig_labels.items():
                if u_term in value:
                    to_append_as_category = key
                    break

            if to_append_as_category != 'O':
                unknown_categories.append(to_append_as_category)
                vec_unknown_term = embedder.get_word_vector(u_term)
                vec_unknown_terms.append(vec_unknown_term)
                tmp_unknown_terms.append(u_term)

        vec_unknown_terms = preprocessing.normalize(vec_unknown_terms)
        u_ar = []
        for u_npar in vec_unknown_terms:
            u_ar.append(u_npar)

        all_vectors = np.concatenate([vectorized_terms, vec_unknown_terms], axis=0)

        """
        AffinityPropagation is not used - it does not converge, and if its converged there is no guarantee that it 
        will converge to enough values. So the pred list has a different length compared to the correct ones.
        """
        for algorithm in ['kMeans', 'hierarchical', 'affinity']:
            print(f"Running {algorithm}")
            for kwargs in KEYWORD_ARGUMENTS[algorithm]:
                failed = False
                cluster_map = pd.DataFrame({
                    'word': tmp_terms + tmp_unknown_terms,
                    'vector': ar + u_ar,
                    'category': categories + ['O' for _ in unknown_categories]
                })

                if algorithm == 'kMeans':
                    algo = KMeans(n_clusters=len(orig_labels), **kwargs)
                elif algorithm == 'hierarchical':
                    algo = AgglomerativeClustering(n_clusters=len(orig_labels), **kwargs)
                else:
                    algo = AffinityPropagation(**kwargs)

                algo.fit(all_vectors)
                if algorithm == 'affinity':
                    if algo.cluster_centers_.shape[0] == 0:
                        failed = True

                if not failed:
                    cluster_map['clusters'] = algo.labels_
                    amount_of_clusters = len(set(algo.labels_))
                    clusters = []
                    representation = []
                    # for i in range(len(orig_labels)):
                    for i in range(amount_of_clusters):
                        cluster = cluster_map[cluster_map['clusters'] == i]

                        if cluster.shape[0] == 1:
                            representation.append('O')
                        else:
                            tmp = cluster[cluster['category'] != 'O']
                            cat = tmp['category'].to_list()
                            rep = max(cat, key=cat.count)
                            for _ in cluster['category']:
                                representation.append(rep)
                    cluster_map['rep'] = representation

                    # unk = cluster_map[cluster_map['category'] == 'O']
                    unk = cluster_map[cluster_map['category'] != 'O']
                    pred = cluster_map['rep'].to_list()
                    true_cat = unk['category'].to_list() + unknown_categories
                    f1 = f1_score(true_cat, pred, average='weighted')
                    hcv_score = homogeneity_completeness_v_measure(true_cat, pred)

                    print(f"HCV Score: {hcv_score}")
                    print(f"F1 Score: {f1}")
            print()
