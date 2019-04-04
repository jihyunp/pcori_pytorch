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
                 dropout: float = None,
                 label_smoothing: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.num_labels = self.vocab.get_vocab_size(label_namespace)
        self.inner_encoder = inner_encoder
        self.label_projection_layer = TimeDistributed(Linear(
            in_features=inner_encoder.get_output_dim(),
            out_features=self.num_labels))

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._label_smoothing = label_smoothing

        self.metrics = {"accuracy": CategoricalAccuracy()}
        # for emotion
        #                "f1_neg": F1Measure(0),
        #                "f1_pos": F1Measure(2)}

        check_dimensions_match(text_field_embedder.get_output_dim(),
                               inner_encoder.get_input_dim(),
                               'text field embedding dim',
                               'inner encoder input dim')
        initializer(self)


    @overrides
    def forward(self,
                utterances: Dict[str, torch.LongTensor],
                speakers: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        inner_mask = util.get_text_field_mask(utterances, 1)
        batch_size, n_utterances, n_words = inner_mask.shape
        outer_mask = util.get_text_field_mask(utterances)

        # Apply embedding
        embedded_utterances = self.text_field_embedder(utterances)
        # Apply dropout
        if self.dropout:
            embedded_utterances = self.dropout(embedded_utterances)

        # Flatten to get inner encodings
        embedded_utterances = embedded_utterances.view(batch_size * n_utterances,
                                                       n_words,
                                                       -1)  # embedding dimension
        inner_mask_flat = inner_mask.view(batch_size * n_utterances, n_words)
        encoded = self.inner_encoder(embedded_utterances, inner_mask_flat)  # (batch*n_utter, lstmout_dim)

        # Unflatten
        encoded = encoded.view(batch_size, n_utterances, -1)  # (batch, n_utter, lstm_out_dim)

        logits = self.label_projection_layer(encoded)
        reshaped_log_probs = logits.view(-1, self.num_labels)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          n_utterances,
                                                                          self.num_labels])
        output = {'logits': logits, 'class_probabilities': class_probabilities,
                  'inner_mask': inner_mask, 'outer_mask': outer_mask}


        if labels is not None:
            loss = util.sequence_cross_entropy_with_logits(logits,
                                                           labels,
                                                           outer_mask,
                                                           label_smoothing=self._label_smoothing)
            for metric in self.metrics.values():
                metric(logits, labels, outer_mask.float())
            output['loss'] = loss

        if metadata is not None:
            output['metadata'] = metadata

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
         return {metric_name: metric.get_metric(reset)
                 for metric_name, metric in self.metrics.items()}
        # return {'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
        #         'f1_neg': self.metrics['f1_neg'].get_metric(reset=reset)[2],
        #         'f1_pos': self.metrics['f1_pos'].get_metric(reset=reset)[2]}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'IndependentRNN':
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        inner_encoder = Seq2VecEncoder.from_params(params.pop('inner_encoder'))
        label_namespace = params.pop('label_namespace', 'labels')
        dropout = params.pop('dropout', None)
        label_smoothing = params.pop('label_smoothing', None)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   inner_encoder=inner_encoder,
                   label_namespace=label_namespace,
                   dropout=dropout,
                   label_smoothing=label_smoothing,
                   initializer=initializer,
                   regularizer=regularizer)


