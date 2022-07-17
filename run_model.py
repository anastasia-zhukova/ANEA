from argparse import Namespace
import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas
import numpy as np

from config import EXEC_RES_PATH, logger
from machine_learning.evaluation import Evaluator
from reading.reader_json import JsonReader

from machine_learning.data_provider import DataProvider
from machine_learning.trainer import Trainer
from machine_learning.argparser import parse_test_arguments

def create_results_df(model_scores: Dict, average: str, model_name:str, current_ts: str, topic: str, random_seed: int, out_domain_datasets: List[str], limit: int, trainsize: float, testsize: float, samples_per_step: List[int], ft_steps: int, learning_rate: float, lr_stepsize: int, epochs: int, trials: int, batchsize: int, best_train_accs: np.ndarray, anea_date: str):
    # DF to save all gridsearch results
    df_cols = ["experiment_datetime", "modelname", "topic", "ANEA_date_used", "out_of_domain_datasets", "HF_limit", "trainsize", "testsize", "out_samples_per_ft_step", "finetuning_steps", "batchsize", "random_seed", "epochs", "trials", "learning_rate", "lr_decay_stepsize", "metric_average", "train_acc", "test_accuracy", "balanced_acc", "F1", "precision", "recall", "MCC"]
    data = [current_ts, model_name, topic, anea_date, out_domain_datasets, limit, trainsize, testsize, samples_per_step, ft_steps, batchsize, random_seed, epochs, trials, learning_rate, lr_stepsize, average, best_train_accs, model_scores['accuracy'], model_scores["acc_balanced"], model_scores["f1"], model_scores["precision"], model_scores["recall"], model_scores["matthews_correlation"]]
    return pandas.DataFrame(data=[data], columns=df_cols)

def save_results_to_file(results_df: pandas.DataFrame, as_json: bool, date_folder: str, topic: str):
    # create results folder
    results_folder = os.path.join(date_folder, topic, "ML_classifier_results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # save results to file
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_results")
    filepath = Path(results_folder, current_datetime)
    if as_json:
        with open(filepath.with_suffix(".json"), "w") as result_file:
            results_df.to_json(result_file)
    else:
        with open(filepath.with_suffix(".csv"), "w") as result_file:
            results_df.to_csv(result_file)


def simple_gridsearch(args: Namespace, dataprovider: DataProvider, in_domain_data: Dict[str, Any], out_domain_data: Dict[str, Any], classes_dict: Dict[str, int], label_dist: np.ndarray, num_knowledge_classes: int, use_knowledge: bool, topic: str, anea_date: str, show_prediction: bool = False):
    """Perform a simple gridsearch to test different hyperparameter combinations without any constraints.

    :param Namespace args: CLI args
    :param DataProvider dataprovider: dataprovider
    :param Dict[str, Any] in_domain_data: domain data samples
    :param Dict[str, Any] out_domain_data: general data samples
    :param Dict[str, int] classes_dict: classes mapped to integer
    :param np.ndarray label_dist: distribution of labels for weighting
    :param int num_knowledge_classes: number of classes in knowledge base
    :param bool use_knowledge: flag whether to use external knowledge
    :param str topic: ANEA topic
    :param str anea_date: date of ANEA execution
    :param bool show_prediction: show model prediction for each training/test sample, defaults to False
    :return pandas.DataFrame: df containing results for each iteration
    """
    
    results_df_list = []
    out_domain_dataset_names = {"anea": args.anea_datasets, "HF": args.hf_datasets, "hf_own": args.own_hf_dataset}
    for random_seed in args.random_seed:
        for ft_steps in args.ft_steps:
            # split data into train/test and mix training data with out of domain data
            train_data, test_data, leftover_data = dataprovider.create_train_test_split(in_domain_data["input"], train_size=args.trainset_size, test_size=args.testset_size, seed=random_seed)
            logger.info(f"Data splits created. Train: {len(train_data)}, test: {len(test_data)}, other: {len(leftover_data)}")
            train_data, num_ft_samples = dataprovider.create_mixed_training_samples(train_data, out_domain_data, ft_steps, random_seed)
            all_train_dataloaders, dataloader_test = dataprovider.create_dataloaders(train_data, test_data, in_domain_data, out_domain_data, args.batchsize)
            trainer = Trainer()
            # add one class because we added class 0 as 'not an entity'
            num_ner_classes = len(classes_dict) + 1
            for learning_rate in args.learning_rate:
                for step_size in args.step_size:
                    for num_epochs in args.num_epochs:
                        # train one/multiple models and return the best (on average)
                        logger.info("==="*100)
                        logger.info(f"Training with following parameters: Seed: {random_seed}, LR: {learning_rate}, step size: {step_size}, epochs: {num_epochs}")
                        logger.info("==="*100)
                        best_train_accs, best_model_state_dict= trainer.train_models(num_ner_classes, all_train_dataloaders, label_dist, num_knowledge_classes, use_knowledge, show_prediction, args.trials, num_epochs, learning_rate, step_size)
                        logger.info(f"Best training accuracy: {best_train_accs}")
                        for average in args.average:
                            # evaluate the model with the best training accuracy
                            logger.info(f"Evaluation average: '{average}'")
                            evaluator = Evaluator(batch_size=args.batchsize)
                            model_scores, model_name = evaluator.evaluate_models(trainer, num_ner_classes, dataloader_test, best_model_state_dict, num_knowledge_classes, use_knowledge, show_prediction, average)
                            # save results in a dataframe
                            current_ts = datetime.datetime.now().strftime("%Y-%m-%d_%R")
                            results_df_list.append(create_results_df(model_scores, average, model_name, current_ts, topic, random_seed, out_domain_dataset_names, args.hf_limit, args.trainset_size, args.testset_size, num_ft_samples, ft_steps, learning_rate, step_size, num_epochs, args.trials, args.batchsize, best_train_accs, anea_date))
    # merge all search results into one big dataframe
    return pandas.concat(results_df_list)

def run(args: Namespace):
    # check for topic CLI argument
    if args.topic is None:
        topic = input("Please enter a topic:\n")
    else:
        topic = args.topic
    # search and read json file for a topic
    try:
        reader = JsonReader(f"{topic}.json")
        raw_text = reader.text_collection
    except:
        logger.error(f"Could not read JSON file for the given topic '{topic}'. Please run ANEA first.")
        sys.exit()
    # check for date CLI argument
    if args.date is None:
        input_date = input("Please enter the date (i.e. 2022-07-05 for 5th July 2022) of the ANEA run you want to use or hit enter to use the current date:\n")
    else: 
        input_date = args.date
    # search and read final_categories.json for a topic and specific date
    if input_date != "":
        date_ = datetime.datetime.strptime(input_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    else:
        now = datetime.datetime.now()
        date_ = now.strftime("%Y-%m-%d")
    date_folder = os.path.join(EXEC_RES_PATH, date_)
    classes_dict_path = os.path.join(date_folder, topic, "final_categories.json")
    # load the class dictionary based on the ANEA output
    if not os.path.exists(classes_dict_path):
        logger.error(f"Could not find a final_categories file at {classes_dict_path}. Please make sure ANEA created such a file.")
        sys.exit()
    else:
        logger.info(f"final_categories file found for {topic} from {date_}. Loading {classes_dict_path}")
        with open(classes_dict_path, "r") as file:
            classes_dict = json.load(file)

    # load additional topics for general domain data 
    general_texts = []
    for t in args.anea_datasets:
        if t != topic:
            reader = JsonReader(f"{t}.json")
            general_texts += reader.text_collection
    # load data from HuggingFace as additional general data
    dataprovider = DataProvider(classes_dict, topic=topic)
    if "News" in args.hf_datasets:
        general_texts += dataprovider.prepare_HF_data("news_commentary", "de-es", limit=args.hf_limit, colname_data="translation", lang="de")
    if "Books" in args.hf_datasets:
        general_texts += dataprovider.prepare_HF_data("opus_books", "de-en", limit=args.hf_limit, lang="de", colname_data="translation")
    if "Amazon" in args.hf_datasets:
        general_texts += dataprovider.prepare_HF_data("amazon_reviews_multi", "de", limit=args.hf_limit, colname_data="review_body")
    # user selected HF dataset
    if args.own_hf_dataset:
        general_texts += dataprovider.prepare_HF_data(dataset_path=args.own_hf_dataset[0], dataset_name=args.own_hf_dataset[1], split=args.own_hf_dataset[2], colname_data=args.own_hf_dataset[3], limit=args.own_hf_dataset[5], shuffle=args.own_hf_dataset[6], seed=args.random_seed, lang=args.own_hf_dataset[4])
    
    # prepare data
    tokenized_in_domain, label_dist, tokenized_out_domain, label_dist_out = dataprovider.preprocess(args.batchsize, raw_text, general_texts)
    # in domain embeddings
    tokenized_in_domain = dataprovider.create_word_embeddings(tokenized_in_domain, args.batchsize)
    input_data_in, labels_in, masks_in = dataprovider.merge_docs(tokenized_in_domain)
    in_domain_data = {"input": input_data_in, "labels": labels_in, "masks": masks_in}  
    # general embeddings
    tokenized_out_domain = dataprovider.create_word_embeddings(tokenized_out_domain, args.batchsize)
    input_data_out, labels_out, masks_out = dataprovider.merge_docs(tokenized_out_domain, len(tokenized_in_domain))
    out_domain_data = {"input": input_data_out, "labels": labels_out, "masks": masks_out}  
    # use label distribution as weights only if specified by user
    if not args.label_dist:
        label_dist=None
    # start gridsearch
    results_df = simple_gridsearch(args, dataprovider, in_domain_data, out_domain_data, classes_dict, label_dist, num_knowledge_classes=0, use_knowledge=False, topic=topic, anea_date=date_)
    save_results_to_file(results_df, args.as_json, date_folder, topic)


if __name__ == "__main__":
    args = parse_test_arguments()
    run(args)