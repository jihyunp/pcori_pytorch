"""
RNN within the utterance (word-level RNN).
"""


from typing import Dict, Optional, List, Any

import logging

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Embedding, TextFieldEmbedder, Seq2VecEncoder, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util

from overrides import overrides


@Model.register('independent_rnn')
class IndependentRNN(Model):
    """
    Parameters
    ----------

    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 inner_encoder: Seq2VecEncoder,
                 label_namespace: str = "labels",
                 emb_dropout: float = None,
                 fc_dropout: float = None,
                 label_smoothing: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.num_labels = self.vocab.get_vocab_size(label_namespace)
        self.inner_encoder = inner_encoder
        self.label_projection_layer = Linear(in_features=inner_encoder.get_output_dim(),
                                             out_features=self.num_labels)
        if emb_dropout:
            self.emb_dropout = torch.nn.Dropout(emb_dropout)
        else:
            self.emb_dropout = None

        if fc_dropout:
            self.fc_dropout = torch.nn.Dropout(fc_dropout)
        else:
            self.fc_dropout = None

        self._label_smoothing = label_smoothing

        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()
        self.metrics = {"accuracy": self._accuracy}
        check_dimensions_match(text_field_embedder.get_output_dim(),
                               inner_encoder.get_input_dim(),
                               'text field embedding dim',
                               'inner encoder input dim')
        initializer(self)

    @overrides
    def forward(self,
                utterance: Dict[str, torch.LongTensor],
                speaker: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        mask = util.get_text_field_mask(utterance).float()
        batch_size, n_words = mask.shape

        # Apply embeddings
        embedded_utterances = self.text_field_embedder(utterance) # (batch, emb_dim)

        # Apply emb_dropout
        if self.emb_dropout:
            embedded_utterances = self.emb_dropout(embedded_utterances)

        # Get LSTM Encoded
        encoded = self.inner_encoder(embedded_utterances, mask)  # (batch, lstmout_dim)

        # FC dropout
        if self.fc_dropout:
            encoded = self.fc_dropout(encoded)

        logits = self.label_projection_layer(encoded)  # (batch, num_labels)
        probs = F.softmax(logits, dim=-1)
        predicted = torch.max(logits, -1)[-1]

        output = {'predicted': predicted,
                  'logits': logits, 'class_probabilities': probs,
                  'mask': mask}

        if labels is not None:
            loss = self._loss(logits, labels.view(-1))
            output['loss'] = loss

            self._accuracy(logits, labels.view(-1))

        if metadata is not None:
            output['metadata'] = metadata

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'IndependentRNN':
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        inner_encoder = Seq2VecEncoder.from_params(params.pop('inner_encoder'))
        label_namespace = params.pop('label_namespace', 'labels')
        emb_dropout = params.pop('emb_dropout', None)
        fc_dropout = params.pop('fc_dropout', None)
        label_smoothing = params.pop('label_smoothing', None)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   inner_encoder=inner_encoder,
                   label_namespace=label_namespace,
                   emb_dropout=emb_dropout,
                   fc_dropout=fc_dropout,
                   label_smoothing=label_smoothing,
                   initializer=initializer,
                   regularizer=regularizer)


