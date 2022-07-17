import math
import random
from string import punctuation
from typing import Dict, List, Tuple, Union

import ftfy
import numpy as np
import spacy
import torch
from config import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from utils.wordvectors import get_model


class OwnDataset(Dataset):
    """A dataset class which can contain our original dataset as well as the concattenated k-hot encoded knowledge
    vectors."""

    def __init__(self, input_data, labels, masks, knowledge=None) -> None:
        self.data = []
        if knowledge is not None:
            for k, v in input_data.items():
                self.data.append(
                    {
                        "data": v,
                        "knowledge": knowledge[k],
                        "labels": labels[k],
                        "mask": masks[k],
                    }
                )
        else:
            for k, v in input_data.items():
                self.data.append({"data": v, "labels": labels[k], "mask": masks[k]})

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataProvider:
    """Provider class to convert ANEA output + Wikidata and prepare a Dataset for model training and evaluation."""

    def __init__(self, classes_dict, topic) -> None:
        self.classes_dict = classes_dict
        self.topic = topic
        self.max_seq_length = 0

    def prepare_HF_data(
        self,
        dataset_path: str,
        dataset_name: str,
        split: str = "train",
        colname_data: str = "text",
        limit: int = None,
        shuffle: bool = False,
        seed: int = None,
        lang: str = None,
    ):
        """(Down-)Load and select a specific dataset and split from HuggingFaces Datasets.

        :param str dataset_path: path of dataset, see HuggingFaces documentation for further information
        :param str dataset_name: name of dataset
        :param str split: name of a split in the dataset, defaults to "train"
        :param str colname_data: name of the column that conatins the required data, defaults to "text"
        :param int limit: maximum number of samples that will be used from the dataset, defaults to None
        :param bool shuffle: whether the dataset should be shuffled before drawing samples from it, defaults to False
        :param int seed: random seed for shuffling the dataset, defaults to None
        :param str lang: if the data column contains multiple language strings, only select those string of the given language, defaults to None
        :return List[str]: texts from the selected dataset
        """
        # load dataset from HF
        dataset_raw = load_dataset(dataset_path, dataset_name)
        # get only relevant split
        dataset = self.get_dataset_split(dataset_raw, split, limit, shuffle, seed)
        texts = dataset[colname_data]
        # get only strings of desired language if it is a translation based dataset
        if lang is not None:
            texts = [text[lang] for text in dataset[colname_data]]
        return texts

    def get_dataset_split(
        self, dataset_raw, splitname="train", size=None, shuffle=True, random_seed=None
    ):
        """Get only the data of one split of the HuggingFace Dataset. Can be shuffeled and limited in size.

        :param splitname: name of the split (must be existing in the HuggingFace dataset)
        :param size: sample size, load all if this is None
        :param shuffle: whether to shuffle the dataset before selecting samples
        :param random_seed: seed for reproducability"""
        # shuffle dataset before selecting samples
        if shuffle:
            dataset = dataset_raw.shuffle(seed=random_seed)
        else:
            dataset = dataset_raw
        # limit dataset to size
        if size != None:
            dataset = dataset[splitname].select(range(size))
        else:
            dataset = dataset[splitname]
        return dataset

    def create_DNER_tags(self) -> Dict[str, int]:
        """Create domain specific NER tags based on the final_categories.json from ANEA

        :return Dict[str, int]: Each class and its according value
        """

        # init with standard tag for 'not an entity'
        ner_tags = {"None": 0}
        for i, entity_class in enumerate(self.classes_dict, start=1):
            ner_tags[entity_class] = i
        return ner_tags

    def create_IOB_tags(self) -> dict:
        """Creates domain specific tags in IOB format. Currently not in use.

        :return dict: IOB tags for each domain class
        """
        # init with standard O tag for 'not an entity'
        ner_tags = {"O": 0}
        # add I and B prefix tags per class
        i = 1
        for entity_class in self.classes_dict:
            ner_tags[f"B-{entity_class}"] = i
            i += 1
            ner_tags[f"I-{entity_class}"] = i
            i += 1
        return ner_tags

    def _tokenize_text(
        self, text: List[str], start_id: int = 0
    ) -> List[Dict[str, Union[int, str, Dict[int, List[str]], None]]]:
        """Convert the text into a SpaCy document and save each sentence as a list of tokens. This also sets the max sequence length, so calling multiple times with different texts should be done consecutively.

        :param List[str] text: Texts that shold be tokenized.
        :param int start_id: Sets a different start ID for the out of domain documents
        :return List[Dict[str, Union[int, str, Dict[int, List[str]], None]]]: List of tokenized documents with all their information
        """
        ger_nlp = spacy.load("de_core_news_sm")
        ger_nlp.enable_pipe("senter")
        tokenized_docs = []
        for text_num, text in enumerate(text):
            doc_id = text_num + start_id
            doc = ger_nlp(ftfy.fix_text(text))
            sentences = {}
            sent_id = 0
            for sentence in doc.sents:
                # ignore punctuation except for '-'
                ignore_punctuation = [p for p in punctuation if p != "-"]
                # also ignore spaces and newlines
                tokenized_sent = [
                    token.text
                    for token in sentence
                    if token.text not in ignore_punctuation and not token.is_space
                ]
                if tokenized_sent:
                    sentences[sent_id] = tokenized_sent
                    sent_id += 1
                # is it possible to adjust the input layer depending on the max sentencen length?
                if self.max_seq_length < len(sentence):
                    self.max_seq_length = len(sentence)
            tokenized_docs.append(
                {
                    "id": doc_id,
                    "text": doc.text,
                    "tokens_per_sentence": sentences,
                    "ner_tags": None,
                }
            )
        return tokenized_docs

    def _tag_tokenized_docs(self, tokenized_docs, ner_tags, batchsize):
        """
        Assign a label for each token in the input documents, based on the ANEA output.
        """
        # total number of tokens annotated, including 'None' annotations
        total_annotations = 0
        # total number of meaningful annotaions, entities that are not 'None'
        entity_annotations = 0
        # class distribution (since this is a weight for the loss function, it should ideally only represent the training data, not all data)
        class_dist = np.zeros(len(ner_tags))
        for doc in tokenized_docs:
            tags_per_sentence = {}
            for i, sentence in doc["tokens_per_sentence"].items():
                # pad labels to uniform size for batched training
                if batchsize > 1:
                    labels = np.zeros(self.max_seq_length)
                else:
                    labels = np.zeros(len(sentence))
                for token_pos, token in enumerate(sentence):
                    token_tag = self._tag_token_as_class(token, ner_tags)
                    labels[token_pos] = token_tag
                    total_annotations += 1
                    if token_tag != ner_tags["None"]:
                        entity_annotations += 1
                    # counter for class distribution
                    class_dist[token_tag] += 1
                tags_per_sentence[i] = labels
            doc["ner_tags"] = tags_per_sentence
        class_dist = class_dist
        return total_annotations, entity_annotations, class_dist

    def _tag_token_as_class(self, token, ner_tags):
        """
        Search for a token word in the AENA output and tag this token with the according class or label it None if its not found.
        """
        for ent_class, ent_class_tokens in self.classes_dict.items():
            if token in ent_class_tokens:
                return ner_tags[ent_class]
        return ner_tags["None"]

    def preprocess(
        self, batchsize: int, domain_texts: List[str], general_texts: List[str] = None
    ) -> Tuple[Dict, np.ndarray, Dict, np.ndarray]:
        """Tokenize data and assign the classes from the ANEA output to the data.

        :param int batchsize: batchsize for training/testing
        :param List[str] domain_texts: list of domain specific texts as they are saved i.e. by ANEA
        :param List[str] general_texts: general domain texts, defaults to None
        :return Tuple[Dict, np.ndarray, Dict, np.ndarray]: tokenized in- and out-of-domain data and the label distribution for each
        """
        logger.info("Begin Preprocessing...")
        ner_tags = self.create_DNER_tags()
        logger.info(f"{len(ner_tags)} domain tags created.")
        # first tokenize domain data
        tokenized_in_domain = self._tokenize_text(domain_texts)
        # then tokenize general data
        if general_texts is not None:
            # has to called before any call to _tag_tokenized_docs because it might chance the max sequence length again
            tokenized_out_domain = self._tokenize_text(
                general_texts, len(tokenized_in_domain)
            )
            (
                total_annotations_out,
                anea_annotations_out,
                label_dist_out,
            ) = self._tag_tokenized_docs(tokenized_out_domain, ner_tags, batchsize)
        else:
            tokenized_out_domain = []
            total_annotations_out = 0
            anea_annotations_out = 0
            label_dist_out = []
        total_annotations, anea_annotations, label_dist = self._tag_tokenized_docs(
            tokenized_in_domain, ner_tags, batchsize
        )
        logger.info(
            f"{len(tokenized_in_domain)} in-domain documents tokenized. {len(tokenized_out_domain)} out-of-domain documents tokenized. "
        )
        logger.info(
            f"In-domain: total of {total_annotations} tokens annotated, {anea_annotations} entities annotated."
        )
        logger.info(
            f"In-domain label distribution: {label_dist.astype(np.int64, copy=True)}"
        )
        logger.info(
            f"Out-of-domain: total of {total_annotations_out} tokens annotated, {anea_annotations_out} entities annotated."
        )
        logger.info(
            f"Out-of-domain label distribution: {label_dist_out.astype(np.int64, copy=True)}"
        )
        return (
            tokenized_in_domain,
            label_dist / np.sum(label_dist),
            tokenized_out_domain,
            label_dist_out / np.sum(label_dist),
        )

    def _embed_sentence(self, sentence_tokens: List[str], model):
        """Get the fastText embeddings for each token in the given sentence.

        :param List[str] sentence_tokens: Sentence as a list of token
        :param _type_ model: fastText model from Gensim
        :return _type_: _description_
        """
        word_vectors = []
        for token in sentence_tokens:
            word_vectors.append(model.get_vector(token))
        return word_vectors

    def create_word_embeddings(self, tokenized_docs, batchsize: int):
        embedding_model = get_model()
        for doc in tokenized_docs:
            sentence_tensors = {}
            sentence_mask = {}
            for k, sentence in doc["tokens_per_sentence"].items():
                word_vectors = []
                # pad sequences to one size if batches should be used
                if batchsize > 1:
                    mask = np.ones(self.max_seq_length, dtype=np.bool8)
                else:
                    mask = np.ones(len(sentence), dtype=np.bool8)
                for token in sentence:
                    # copy because numpy array from get_vector is not writable, but needs to be for pytorch
                    word_vectors.append(
                        torch.from_numpy(embedding_model.get_vector(token).copy())
                    )
                if batchsize > 1:
                    # append 0 filled tensors so that all tensors have the same length
                    for i in range(len(sentence), self.max_seq_length):
                        word_vectors.append(torch.zeros_like(word_vectors[0]))
                        # mask filled tensors as unimportant
                        mask[i] = 0
                # stack word vectors to single sentence tensor as input sequence
                sentence_tensors[k] = torch.stack(tuple(word_vectors), dim=0)
                sentence_mask[k] = torch.from_numpy(mask)
            doc["sentence_tensors"] = sentence_tensors
            doc["sentence_mask"] = sentence_mask
        return tokenized_docs

    def merge_docs(self, tokenized_docs, start_id: int = 0):
        merged_tensors = {}
        merged_labels = {}
        merged_masks = {}
        for doc_id, doc in enumerate(tokenized_docs):
            for sent_id, tensor in doc["sentence_tensors"].items():
                unique_id = f"{doc_id + start_id}-{sent_id}"
                merged_tensors[unique_id] = tensor
                merged_labels[unique_id] = doc["ner_tags"][sent_id]
                merged_masks[unique_id] = doc["sentence_mask"][sent_id]
        return merged_tensors, merged_labels, merged_masks

    def create_train_test_split(
        self, tensors: dict, train_size: float, test_size: float, seed: int = None
    ):
        """Split the data into training, testing and leftover data depending on the train and test size. It is sufficient to only split the input tensors, because OwnDataset will get the labels and masks for each tensor according to the unique ID.

        :param dict tensors: Our input tensors for the model. These are split randomly.
        :param float train_size: Relative size of training data, value in [0,1]
        :param float test_size: Relative size of test data, value in [0,1]
        """
        total_samples = len(tensors)
        num_training_samples = math.floor(total_samples * train_size)
        num_test_samples = math.floor(total_samples * test_size)
        # point where the train and test split should end
        end_test_split = num_training_samples + num_test_samples
        training_samples = {}
        test_samples = {}
        leftover_samples = {}
        if seed is not None:
            random.seed(seed)
        # sample dictionary keys as random list = shuffling
        random_list = random.sample(tensors.keys(), total_samples)
        for i in random_list[:num_training_samples]:
            training_samples[i] = tensors[i]
        for i in random_list[num_training_samples:end_test_split]:
            test_samples[i] = tensors[i]
        for i in random_list[end_test_split:]:
            leftover_samples[i] = tensors[i]
        return training_samples, test_samples, leftover_samples

    def create_mixed_training_samples(
        self,
        training_samples: Dict,
        out_domain_data: Dict,
        ft_steps: int,
        seed: int = None,
    ) -> Tuple[List[Dict], List[int]]:
        """Mix out-of-domain data with the in-domain training data. Number of out-of-domain samples used depend on the current fine tuning step.

        :param Dict training_samples: in-domain samples
        :param Dict out_domain_data: out-of-domain samples
        :param int ft_steps: number of fine-tuning steps that use out-of-domain data
        :param int seed: random seed for sampling, defaults to None
        :return List[Dict], List[int]: mixed samples and how many out-of-domain samples were used for each fine-tuning step
        """
        if not out_domain_data:
            return [training_samples]
        if seed is not None:
            random.seed(seed)
        # compute how many out of domain samples are mixed in traning data per iteration
        num_ft_samples = [
            math.floor(1 / ft * len(out_domain_data["input"]))
            for ft in range(1, ft_steps + 1)
        ]
        logger.info(
            f"Out-of-domain samples used for the fine-tuning steps: {num_ft_samples}"
        )
        mixed_samples_all_ft_steps = []
        for num_sample in num_ft_samples:
            mixed_samples = {k: v.detach().clone() for k, v in training_samples.items()}
            out_samples = random.sample(out_domain_data["input"].keys(), num_sample)
            for k in out_samples:
                mixed_samples[k] = out_domain_data["input"].get(k).detach().clone()
            mixed_samples_all_ft_steps.append(mixed_samples)
        # for the last fine tuning step use only in-domain data
        mixed_samples_all_ft_steps.append(training_samples)
        return mixed_samples_all_ft_steps, num_ft_samples

    def create_dataloaders(
        self,
        training_samples: List[Dict],
        test_samples: Dict,
        in_domain_data: Dict,
        out_domain_data: Dict,
        batchsize: int,
    ) -> Tuple[List[DataLoader], DataLoader]:
        """Convert the dictionaries that contain our data into DataLoaders that handle batch loading for training and evaluation.

        :param List[Dict] training_samples: different mixtures of embedded in-domain and out-of-domain samples
        :param Dict test_samples: test samples
        :param Dict in_domain_data: raw (not embedded) domain data
        :param Dict out_domain_data: raw general data
        :param int batchsize: batchsize
        :return Tuple[List[DataLoader], DataLoader]: DataLoaders prepared for training/testing
        """
        # merge in and out of domain data so the mixed Dataset can get correct labels and masks
        all_data = {}
        for k in in_domain_data:
            all_data[k] = {**out_domain_data[k], **in_domain_data[k]}
        all_train_dataloaders = []
        # create a dataloader for each fine tuning step
        for train_sample in training_samples:
            dataset_train = OwnDataset(
                train_sample, all_data["labels"], all_data["masks"]
            )
            all_train_dataloaders.append(
                DataLoader(dataset=dataset_train, batch_size=batchsize)
            )
        dataset_test = OwnDataset(
            test_samples, in_domain_data["labels"], in_domain_data["masks"]
        )
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batchsize)
        return all_train_dataloaders, dataloader_test
