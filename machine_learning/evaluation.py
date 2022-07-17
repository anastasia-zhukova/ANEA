from datasets import load_metric
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from config import logger
from machine_learning.MEDIC_approach import BiLSTM
from machine_learning.trainer import Trainer


class Evaluator:
    def __init__(self, size=25, batch_size=4, num_epochs=3) -> None:
        self.size = size
        self.batch_size = batch_size
        # set device to cuda if possible
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.num_epochs = num_epochs

    def evaluation_loop(self, model, dataloader, average, show_prediction=False):
        """Evaluate the given model on the test data and return a dictionary of different scores:
        - accuracy
        - f1
        - recall
        - precision
        - matthews_correlation
        This also prints the scores and if show_prediction is True, also the predictions of the model.
        """
        all_targets = None
        final_scores = {}
        # set model to evaluation mode
        model.eval()
        # load metrics
        accuracy_metric = load_metric("accuracy")
        f1_metric = load_metric("f1")
        mcc_metric = load_metric("matthews_correlation")
        recall_metric = load_metric("recall")
        precision_metric = load_metric("precision")
        # disable gradient changes to speed up, since we do not want to learn in the evaluation anyway
        with torch.no_grad():
            # save predictions as list, to print as one block in the end
            if show_prediction:
                epoch_predictions = []
                epoch_targets = []
            total_correct_preditions = 0
            total_predictions = 0
            for batch in tqdm(dataloader):
                # predict test data with our model
                logits, predictions = model(batch)
                # get targets to device
                targets = batch.get("labels").to(self.device)
                # target mask
                mask = batch.get("mask").to("cuda")
                if show_prediction:
                    epoch_predictions.append(predictions)
                    epoch_targets.append(targets)
                # concatenate to evaluate all predictions
                if all_targets is None:
                    all_targets = targets
                    all_predictions = predictions
                    all_masks = mask.float()
                else:
                    # reduce to 1D tensor before concatenation since we need a 1D array for score calculation anyway and this allows us to use inputs with different sequence lengths
                    all_targets = torch.cat(
                        (all_targets.ravel(), targets.ravel()), dim=0
                    )
                    all_predictions = torch.cat(
                        (all_predictions.ravel(), predictions.ravel()), dim=0
                    )
                    all_masks = torch.cat(
                        (all_masks.ravel(), mask.float().ravel()), dim=0
                    )
                # add batch to different metrics
                accuracy_metric.add_batch(
                    predictions=predictions.ravel(), references=targets.long().ravel()
                )
                f1_metric.add_batch(
                    predictions=predictions.ravel(), references=targets.ravel()
                )
                mcc_metric.add_batch(
                    predictions=predictions.ravel(), references=targets.ravel()
                )
                recall_metric.add_batch(
                    predictions=predictions.ravel(), references=targets.ravel()
                )
                precision_metric.add_batch(
                    predictions=predictions.ravel(), references=targets.ravel()
                )

                # count correct predictions per batch
                if targets.shape[0] > 1:
                    # all sequences have the same length -> ignore padding via target mask
                    total_correct_preditions += (
                        ((predictions == targets) == mask).sum().item()
                    )
                else:
                    # sequences can have different lengths
                    total_correct_preditions += (predictions == targets).sum().item()
                # total_predictions += pred.size(dim=1)
                total_predictions += mask.sum().item()

        # compute metrics score for all batches
        all_masks = all_masks.to("cpu")
        final_acc = accuracy_metric.compute(sample_weight=all_masks)
        # this was used to see how the mask weighting would effect the accuracy_metric
        # it seems to yield the correct results now
        final_acc_2 = total_correct_preditions / total_predictions
        balanced_acc = balanced_accuracy_score(
            all_targets.ravel().to("cpu").numpy(),
            all_predictions.ravel().to("cpu").numpy(),
            sample_weight=all_masks,
        )
        # set average to macro for f1 and recall, when binary average can not be computed
        if model.get_num_labels() == 2:
            average = "binary"
            final_f1 = f1_metric.compute()
            final_recall = recall_metric.compute()
        else:
            # use selected average and target masks as weights
            final_f1 = f1_metric.compute(average=average, sample_weight=all_masks)
            final_recall = recall_metric.compute(
                average=average, sample_weight=all_masks
            )
            final_precision = precision_metric.compute(
                average=average, sample_weight=all_masks
            )
        final_mcc = mcc_metric.compute(sample_weight=all_masks)
        # compute confusion matrix
        final_scores["confusion_matrix"] = confusion_matrix(
            all_targets.ravel().to("cpu").numpy(),
            all_predictions.ravel().to("cpu").numpy(),
            sample_weight=all_masks,
        )
        # save all scores in a single dictionary
        final_scores["accuracy"] = final_acc["accuracy"]
        final_scores["acc_balanced"] = balanced_acc
        final_scores["f1"] = final_f1["f1"]
        final_scores["recall"] = final_recall["recall"]
        final_scores["precision"] = final_precision["precision"]
        final_scores["matthews_correlation"] = final_mcc["matthews_correlation"]
        # print predictions and model scores
        if show_prediction:
            for pre, tar in zip(epoch_predictions, epoch_targets):
                logger.info(f"\nPredictions: {pre}\nTargets: {tar}")
        logger.info(
            f'Test Scores:\nAccuracy: {final_acc["accuracy"]} Acc. balanced: {final_scores["acc_balanced"]}\nF1: {final_f1["f1"]}\nRecall: {final_recall["recall"]}\nPrecision: {final_scores["precision"]}'
        )
        logger.info(
            f'Matthews correlation coefficient: {final_scores["matthews_correlation"]}'
        )
        logger.info(f'Confusion matrix:\n{final_scores["confusion_matrix"]}')
        return final_scores, average

    def evaluate(self, model, dataloader_test, average, show_prediction=False):
        """Starts model evaluation loop and returns the model scores.
        :returns: dict with scores"""
        logger.info(f'\nEvaluating "{model.get_model_name()}" on chosen dataset:')
        # evaluation using test split
        model_scores, average = self.evaluation_loop(
            model, dataloader_test, show_prediction=show_prediction, average=average
        )
        return model_scores

    def evaluate_models(
        self,
        trainer: Trainer,
        num_ner_classes: int,
        dataloader_test: DataLoader,
        best_model_state_dict: dict,
        num_knowledge_classes: int,
        use_knowledge :bool,
        show_prediction :bool,
        average: str,
    ):
        """Load a model from the given state dict and evaluate it."""
        model = BiLSTM(
            trainer.device,
            num_ner_classes,
            num_knowledge_classes,
            use_knowledge=use_knowledge,
        )
        model.load_state_dict(best_model_state_dict)
        # evaluate model
        return (
            self.evaluate(
                model, dataloader_test, show_prediction=show_prediction, average=average
            ),
            model.get_model_name(),
        )
