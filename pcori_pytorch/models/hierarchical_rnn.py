"""
Neural conditional random field (CRF) tagging model.

Essentially the same as the one in:

    https://github.com/allenai/allennlp/

but adapted to make predictions on sequences of utterances instead of words.  E.g. An additional
encoding step must be performed to obtain a single vector representation of each utterance.
"""

from typing import Dict, Optional, List, Any

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.modules import ConditionalRandomField, Seq2SeqEncoder, Seq2VecEncoder, \
        TimeDistributed, TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from pcori_pytorch.training import FuckingAccuracy


@Model.register('hierarchical_rnn')
class HierarchicalRNN(Model):
    """
    The ``HierarchicalCRF`` encodes sequences of sequences of text and then uses a conditional
    random field (CRF) model to predict a tag for each inner sequence. Representations of each of
    the inner sequence are obtained with a ``Seq2VecEncoder``. These vectors are subsequently fed
    into a ``Seq2SeqEncoder`` to encode the outer sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A vocabulary, needed to compute the sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    inner_encoder : ``Seq2VecEncoder``
        The encoder that will be used to encode inner sequences.
    outer_encoder : ``Seq2SeqEncoder``
        The encoder that will be used to encode outer sequences.
    label_namespace :``str``, optional (default=``labels``)
        Needed to compute the SpanBasedF1Measure metric.
    dropout : ``float``, optional (default=``None``)
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 inner_encoder: Seq2VecEncoder,
                 outer_encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(HierarchicalRNN, self).__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.inner_encoder = inner_encoder
        self.outer_encoder = outer_encoder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.label_projection_layer = TimeDistributed(Linear(outer_encoder.get_output_dim(),
                                                             self.num_tags))
        # self.metrics = {"accuracy": FuckingAccuracy()}
        self.metrics = {"accuracy": CategoricalAccuracy()}
        self._loss = torch.nn.CrossEntropyLoss()

        check_dimensions_match(text_field_embedder.get_output_dim(),
                               inner_encoder.get_input_dim(),
                               'text field embedding dim',
                               'inner encoder input dim')
        check_dimensions_match(inner_encoder.get_output_dim(),
                               outer_encoder.get_input_dim(),
                               'inner encoder output dim',
                               'outer encoder input dim')
        initializer(self)

    @overrides
    def forward(self,
                utterances: Dict[str, torch.LongTensor],
                speakers: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Note: Currently ignores speaker data.

        Parameters
        ----------
        utterances : ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()``, which should be passed directly to a
            ``TextFieldEmbedder``.
        speakers : ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()``, which should be passed directly to a
            ``TextFieldEmbedder``.
        labels: ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor containing the observed class labels.
        metadata : ``List[Dict[str, Any]]``, optional (default = ``None``)
            Metadata about the inputs (e.g. session_id, the original sequences of words, etc.).

        Returns
        -------
        An output dictionary consisting of:

        logits: ``torch.FloatTensor``
            The logits output by the ``tag_projection_layer``.
        mask: ``torch.LongTensor``
            The text field mask for the input utterances.
        """
        # Extract masks - note: there are two, the first is a word level mask (e.g. accounts for
        # padding the sequences of words within utterances), the second is an utterance level mask
        # (e.g. accounts for padding sequences of utterances).
        inner_mask = util.get_text_field_mask(utterances, 1)
        outer_mask = util.get_text_field_mask(utterances)
        batch_size, n_utterances, n_words = inner_mask.shape

        # Apply embedding and flatten to get inner encodings
        embedded_utterances = self.text_field_embedder(utterances)
        if self.dropout:
            embedded_utterances = self.dropout(embedded_utterances)
        # TODO: Replace with TimeDistributed
        embedded_utterances = embedded_utterances.view(batch_size * n_utterances, n_words, -1)
        inner_mask = inner_mask.view(batch_size * n_utterances, n_words)
        inner_encoded = self.inner_encoder(embedded_utterances, inner_mask)
        if self.dropout:
            inner_encoded = self.dropout(inner_encoded)

        # Unflatten and get outer encodings
        inner_encoded = inner_encoded.view(batch_size, n_utterances, -1)
        outer_encoded = self.outer_encoder(inner_encoded, outer_mask)
        # (batch, n_utter, lstm_out_dim)

        if self.dropout:
            outer_encoded = self.dropout(outer_encoded)

        # Project
        logits = self.label_projection_layer(outer_encoded)
        # (batch, n_utter, n_labels)

        reshaped_log_probs = logits.view(-1, self.num_tags)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          n_utterances,
                                                                          self.num_tags])
        # predicted = torch.max(logits, -1)[-1]  # (batch, n_utter)
        output = {'logits': logits, 'class_probabilities': class_probabilities,
                  'inner_mask': inner_mask, 'outer_mask': outer_mask}

        if labels is not None:
            loss = util.sequence_cross_entropy_with_logits(logits,
                                                           labels,
                                                           outer_mask)
            for metric in self.metrics.values():
                metric(logits, labels, outer_mask.float())
            output['loss'] = loss

        if metadata is not None:
            output['metadata'] = metadata

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'HierarchicalRNN':
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        inner_encoder = Seq2VecEncoder.from_params(params.pop('inner_encoder'))
        outer_encoder = Seq2SeqEncoder.from_params(params.pop('outer_encoder'))
        label_namespace = params.pop('label_namespace', 'labels')
        dropout = params.pop('dropout', None)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   inner_encoder=inner_encoder,
                   outer_encoder=outer_encoder,
                   label_namespace=label_namespace,
                   dropout=dropout,
                   initializer=initializer,
                   regularizer=regularizer)

