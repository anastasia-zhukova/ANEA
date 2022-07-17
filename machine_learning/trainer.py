from copy import deepcopy
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import logger
from machine_learning.MEDIC_approach import BiLSTM


class Trainer:
    def __init__(self) -> None:
        # set device to cuda if possible
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def training_loop(
        self,
        model,
        dataloader_train,
        num_epochs,
        learning_rate,
        step_size,
        show_prediction=False,
        label_weights=None,
    ):
        """Trains the model and shows traning accuracy.

        :param model: model to be trained
        :param dataloader_train: prepared dataloader of the training data
        :param num_epochs: training epochs
        :param show_prediction: whether to print the predictions for each batch
        :param label_weights: weights to accord for class imbalance, used in the loss function
        """
        # set model to train on correct device
        model.train().to(self.device)
        # optimizer for training
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
        # learning rate scheduler
        lr_scheduler = StepLR(
            optimizer=optimizer,
            step_size=step_size,
        )
        if label_weights != None:
            # loss function weighted by label distribution in data
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(label_weights).to(self.device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        # total predictions
        total_predictions = 0
        total_correct_preditions = 0
        # training loop
        for epoch in range(num_epochs):
            logger.info("==" * 100)
            logger.info(f"\nEpoch: {epoch}")
            total_loss_last_epoch = 0
            num_batches_last_epoch = 0
            if show_prediction:
                epoch_predictions = []
                epoch_targets = []
            for i, batch in enumerate(tqdm(dataloader_train)):
                num_batches_last_epoch = i + 1
                # feed input batchwise to language model
                logits, pred = model(batch)
                # get targets and masks to device
                targets = batch.get("labels").to(self.device)
                mask = batch["mask"].to(self.device)
                if show_prediction:
                    epoch_predictions.append(pred)
                    epoch_targets.append(targets)
                # compute loss
                loss = criterion(torch.transpose(logits, 1, 2).double(), targets.long())
                total_loss_last_epoch += loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # count correct predictions per batch
                if targets.shape[0] > 1:
                    # all sequences have the same length
                    total_correct_preditions += ((pred == targets) == mask).sum().item()
                else:
                    # sequences can have different lengths
                    total_correct_preditions += (pred == targets).sum().item()
                # total_predictions += pred.size(dim=1)
                total_predictions += mask.sum().item()
            # show average epoch loss
            logger.info(f"avg. loss: {total_loss_last_epoch / num_batches_last_epoch}")
        if show_prediction:
            for pre, tar in zip(epoch_predictions, epoch_targets):
                logger.info(f"\nPredictions: {pre}\nTargets: {tar}")
        # compute training accuracy
        train_accuracy = total_correct_preditions / total_predictions
        logger.info(f"\nTraining Accuracy: {train_accuracy}\n")
        return train_accuracy

    def train_models(
        self,
        num_ner_classes: int,
        all_train_dataloaders: List[DataLoader],
        label_dist: np.ndarray,
        num_knowledge_classes: int,
        use_knowledge: bool,
        show_prediction: bool,
        trials: int,
        num_epochs: int,
        learning_rate: float,
        step_size: int,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Train one or multiple models on the same parameters. Select the model with the best training accuracy (on average).

        :param int num_ner_classes: Number of domain specific classes
        :param List[DataLoader] all_train_dataloaders: Dataloaders with different mixtures of in-domain and out-of-domain data
        :param np.ndarray label_dist: Relative distribution of labels, for weighting
        :param int num_knowledge_classes: Number of classes in the knowledge base
        :param bool use_knowledge: Whether knowledge is used or not
        :param bool show_prediction: Show predictions for each batch during training
        :param int trials: Number of models that will be initialized and trained with the same parameters
        :param int num_epochs: Number of epochs each model is trained
        :param float learning_rate: Learning rate
        :param int step_size: Interval in wich the learning rate will be changed (in epochs)
        :return Tuple[List[float], Dict[str, Any]]: Training accuracy per epoch and state dict of model
        """
        best_train_accs = np.zeros(len(all_train_dataloaders))
        # init and train multiple models
        for trial in range(trials):
            logger.info(f"Training model nr. {trial}")
            model = BiLSTM(
                self.device,
                num_ner_classes,
                num_knowledge_classes,
                use_knowledge=use_knowledge,
            )
            # training accuracy for each fine tuning step
            train_accs = np.zeros(len(all_train_dataloaders))
            # fine tune model while decreasing amount of out-of-domain data
            for ft_num, dataloader in enumerate(all_train_dataloaders):
                logger.info(
                    f"Fine-tuning iteration no. {ft_num+1} on {len(dataloader.dataset.data)} samples"
                )
                train_accs[ft_num] = self.training_loop(
                    model,
                    dataloader,
                    num_epochs,
                    learning_rate,
                    step_size,
                    show_prediction=show_prediction,
                    label_weights=label_dist,
                )
            # use average acc over all finetuning steps to find the best model
            if np.mean(train_accs) > np.mean(best_train_accs):
                best_train_accs = train_accs
                best_model_state_dict = deepcopy(model.state_dict())
        return best_train_accs, best_model_state_dict
