import json

import numpy as np
import spacy
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, \
    precision_recall_fscore_support
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, pipeline

from woe.data_preprocessor import Preprocessor
from datasets import load_metric

metric = load_metric("seqeval")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    o_acc, o_recall, o_prec = 0, 0, 0
    for y_true, y_pred in zip(true_labels, true_predictions):
        acc = accuracy_score(y_true, y_pred)
        prec, recall, _, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        o_acc += acc
        o_recall += recall
        o_prec += prec
    o_acc /= len(true_labels)
    o_recall /= len(true_labels)
    o_prec /= len(true_labels)
    return {
        'accuracy': o_acc,
        'precision': o_prec,
        'recall': o_recall,
        'f1': (2 * o_prec * o_recall) / (o_prec + o_recall)
    }


nlp = spacy.load('de_core_news_sm')
used_model = "distilbert-base-german-cased"


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


if __name__ == '__main__':
    pre = Preprocessor(nlp)
    label_names, train, val = pre.preprocess()
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

    data = {'data': train}
    with open("tests/train.json", "w") as f:
        json.dump(data, f)

    data = {'data': val}
    with open("tests/val.json", "w") as f:
        json.dump(data, f)

    raw_dataset = load_dataset("json", data_files={"train": "tests/train.json", "val": "tests/val.json"}, field="data")

    tokenizer = AutoTokenizer.from_pretrained(used_model)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    tokenized_datasets = raw_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset['train'].column_names
    )

    model = AutoModelForTokenClassification.from_pretrained(
        used_model,
        id2label=pre.id2label,
        label2id=pre.label2id
    )
    model.to(device)

    for param in model.distilbert.parameters():
        param.requires_grad = False

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
    trainer.train()
    # results = trainer.evaluate(tokenized_datasets["val"])
    # test = trainer.predict(tokenized_datasets['val'])

    token_classifier = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    token_classifier.device = torch.device(device)

    texts = pre.get_new_texts()
    all_outs = []
    for text in texts:
        out = token_classifier(text)
        all_outs.append(out)

    results = {}
    for doc_id, outputs in enumerate(all_outs):
        for idx, all_preds in enumerate(outputs):
            if len(all_preds) == 0:
                continue
            highest_score = 0
            best_pred = ""
            for pred in all_preds:
                if pred['score'] > highest_score:
                    best_pred = pred['entity_group']
                    highest_score = pred['score']
            if best_pred in results:
                results[best_pred].append(texts[doc_id][idx])
            else:
                results[best_pred] = [texts[doc_id][idx]]

    with open("ner_prediction_without_freeze.json", "w") as f:
        json.dump(results, f)

    print()
