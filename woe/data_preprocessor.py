import json
import os.path

import ftfy
from tqdm import tqdm

DOC_LIM = None


class Preprocessor:
    def __init__(self, nlp):
        self.nlp = nlp
        self.label2id = None
        self.id2label = None

    def preprocess(self):
        orig_texts = []
        orig_labels = {'O': []}
        known_dir = os.path.join("data", "already_known")

        text_file = os.path.join(known_dir, "databases", "databases_texts.json")
        with open(text_file, "r", encoding='utf8') as f:
            orig_texts += json.load(f)

        label_file = os.path.join(known_dir, "databases", "databases_labels.json")
        with open(label_file, "r", encoding='utf8') as f:
            text_labels = json.load(f)
            for key, value in text_labels.items():
                if key in orig_labels:
                    orig_labels[key] += value
                else:
                    orig_labels[key] = value

        annot_texts = self.apply_spacy(orig_texts)

        label2id = {}
        id2label = {}

        counter = 0
        for key in orig_labels:
            id2label[counter] = key
            label2id[key] = counter
            counter += 1

        self.label2id = label2id
        self.id2label = id2label

        sent_ids = []
        words = []
        labels = []
        test_sent_ids = []
        test_words = []
        test_labels = []
        for i, doc in enumerate(annot_texts):
            doc_sent_ids = []
            doc_words = []
            doc_labels = []
            test_doc_sent_ids = []
            test_doc_words = []
            test_doc_labels = []
            for sent_idx, sent in enumerate(doc.sents):
                for token in sent:
                    if i >= len(annot_texts):    # - 2:
                        test_doc_sent_ids.append(sent_idx)
                        test_doc_words.append(sent_idx)
                    # sent_ids.append(sent_idx)
                    else:
                        doc_sent_ids.append(sent_idx)
                        doc_words.append(token.text)
                    # words.append(token.text)

                    # to_insert_as_label = token.ent_iob_ if token.ent_iob_ == 'O'\
                    #     else f"{token.ent_iob_}-{token.ent_type_}"
                    to_insert_as_label = 'O'
                    for key, value in orig_labels.items():
                        if token.text in value:
                            to_insert_as_label = key
                            break
                    # labels.append(to_insert_as_label)
                    if i >= len(annot_texts):   # - 2:
                        test_doc_labels.append(label2id[to_insert_as_label])
                    else:
                        doc_labels.append(label2id[to_insert_as_label])

            if i >= len(annot_texts):   # - 2:
                test_sent_ids.append(test_doc_sent_ids)
                test_words.append(test_doc_words)
                test_labels.append(test_doc_labels)
            else:
                sent_ids.append(doc_sent_ids)
                words.append(doc_words)
                labels.append(doc_labels)

        train = {
            'sentence_id': sent_ids,
            'tokens': words,
            'ner_tags': labels
        }

        orig_test_texts = []
        with open("data/unknown/Datenbanken.json", "r", encoding='utf8') as f:
            orig_test_texts = json.load(f)

        annot_test_texts = self.apply_spacy(orig_test_texts)

        test_doc_sent_ids = []
        test_doc_words = []
        test_doc_labels = []
        for i, doc in enumerate(annot_test_texts):
            test_words = []
            test_sent_ids = []
            test_labels = []
            for sent_idx, sent in enumerate(doc.sents):
                for token in sent:
                    test_sent_ids.append(sent_idx)
                    test_words.append(token.text)

                    to_insert_as_label = 'O'
                    for key, value in orig_labels.items():
                        if token.text in value:
                            to_insert_as_label = key
                            break
                    test_labels.append(label2id[to_insert_as_label])
            test_doc_sent_ids.append(test_sent_ids)
            test_doc_words.append(test_words)
            test_doc_labels.append(test_labels)

        val = {
            'sentence_id': test_doc_sent_ids,
            'tokens': test_doc_words,
            'ner_tags': test_doc_labels
        }

        return list(set(orig_labels.keys())), train, val

    def get_new_texts(self):
        with open("data/unknown/Datenbanken.json", "r") as f:
            orig_texts = json.load(f)

        # texts = ""
        # for text in orig_texts:
        #     texts += text

        annot_texts = self.apply_spacy(orig_texts)
        texts_as_words = []
        for i, doc in enumerate(annot_texts):
            doc_words = []
            for sent_idx, sent in enumerate(doc.sents):
                for token in sent:
                    doc_words.append(token.text)
            texts_as_words.append(doc_words)
        return texts_as_words

    def apply_spacy(self, orig_texts):
        preprocessed_docs = []
        with tqdm(total=len(orig_texts)) as bar:
            for i, text in enumerate(orig_texts[:DOC_LIM] if DOC_LIM is not None else orig_texts):
                text = text.replace("\n", "")
                doc = self.nlp(ftfy.fix_text(text))
                if doc.doc is not None:
                    preprocessed_docs.append(doc)
                bar.update()
        return preprocessed_docs
