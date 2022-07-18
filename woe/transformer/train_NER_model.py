import json
import os
import pickle

import pandas as pd
import spacy
import torch
from simpletransformers.ner import NERModel, NERArgs

from woe.data_preprocessor import Preprocessor

TASK = "ner"
MODEL_TYPE = "distilbert"   # "xlmroberta"
MODEL_NAME = "distilbert-base-german-cased"  # "xlm-roberta-large-finetuned-conll03-german"
nlp = spacy.load('de_core_news_sm')

if __name__ == '__main__':
    preprocessor = Preprocessor(nlp)
    keys, text_dict = preprocessor.preprocess()
    text_as_dataframe = pd.DataFrame(text_dict)

    args = NERArgs()
    args.labels_list = keys
    args.do_lower_case = False
    args.num_train_epochs = 20
    args.overwrite_output_dir = True
    args.silent = True

    cuda_available = torch.cuda.is_available()

    model = NERModel(
        MODEL_TYPE,
        MODEL_NAME,
        labels='ner_tags',
        args=args,
        use_cuda=cuda_available
    )

    model.train_model(text_as_dataframe, "logs/")

    chosen_file = os.path.join("data", "unknown", "Datenbanken.json")
    with open(chosen_file, "r") as f:
        new_texts = json.load(f)

    preds, model_output = model.predict(new_texts)

    result = {}
    for text in preds:
        for word_as_dict in text:
            for word, predicted_label in word_as_dict.items():
                if predicted_label == 'O':
                    continue
                if predicted_label not in result:
                    result[predicted_label] = [word]
                else:
                    result[predicted_label].append(word)

    os.makedirs(os.path.join("data", "ner_output", MODEL_NAME), exist_ok=True)
    with open(os.path.join("data", "ner_output", MODEL_NAME, f"ner_model_output_for_Datenbanken.json"), "w") as f:
        json.dump(result, f)

    print()
