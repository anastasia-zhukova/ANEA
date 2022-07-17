import torch
import torch.nn as nn
from TorchCRF import CRF
import numpy as np


class BiLSTM(nn.Module):

    MODEL_NAME = "MEDIC_approach"

    def __init__(
        self, device, num_data_classes, num_knowledge_in_features, use_knowledge=True
    ):
        super().__init__()
        self.device = device
        self.num_data_classes = num_data_classes
        # default size of wordvectors in fastText is 300
        self.wordvector_size = 300
        # fastText embeddings + knowledge
        if use_knowledge:
            self.num_inputs = 2
        # fastText embeddings only
        else:
            self.num_inputs = 1
        # knowledge embedding
        self.knowledge_embedding = nn.Linear(
            num_knowledge_in_features, self.wordvector_size
        ).to(self.device)
        # LSTM to learn embeddings
        self.lstm = nn.LSTM(
            self.wordvector_size * self.num_inputs,
            self.wordvector_size * self.num_inputs,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm.to(self.device)
        # crf layer to decode results
        self.crf = CRF(num_data_classes)
        self.crf.to(self.device)
        # linear layer to use before the CRF
        self.dense = nn.Linear(
            # 2 inputs to lstm (fastText and knowledge embedding), 2 (because bidirectional LSTM)
            self.wordvector_size * self.num_inputs * 2,
            self.num_data_classes,
        )
        self.dense.to(self.device)

    def get_num_labels(self):
        return self.num_data_classes

    def get_model_name(self):
        return BiLSTM.MODEL_NAME

    def forward(self, inputs):
        # only fastText embeddings
        lm_inputs = inputs.get("data").to(self.device)
        # check if knowledge should be used
        if self.num_inputs == 2:
            # get knowledge embeddings
            knowledge = inputs.get("knowledge")
            knowledge = knowledge.to(self.device)
            knowledge = self.knowledge_embedding(knowledge.float())
            # concat input embeddings and knowledge embeddings
            data_and_knowledge_embedded = torch.cat((lm_inputs, knowledge), dim=2)
        else:
            data_and_knowledge_embedded = lm_inputs
        # feed embedded inputs to BiLSTM
        lstm_all_hidden, (lstm_last_hidden, lstm_last_cell) = self.lstm(
            data_and_knowledge_embedded
        )
        # dense layer for logits
        logits = self.dense(lstm_all_hidden)
        # CRF to decode sequence labels
        decoded_logits = self.crf.viterbi_decode(
            logits, inputs.get("mask").to(self.device)
        )
        decoded_logits = torch.from_numpy(
            np.array([xi + [0] * (logits.shape[1] - len(xi)) for xi in decoded_logits])
        ).to(self.device)
        return logits, decoded_logits
