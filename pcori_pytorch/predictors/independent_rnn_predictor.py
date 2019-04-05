from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('independent_rnn_predictor')
class IndependentRNNPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        session_id = json_dict['session_id']
        utterances = json_dict['utterance']
        speakers = json_dict['speaker']
        labels = json_dict['label']
        instance = self._dataset_reader.text_to_instance(session_id=session_id,
                                                         utterance=utterances,
                                                         speaker=speakers,
                                                         labels=labels)
        #label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        #all_labels = [label_dict[i] for i in range(len(label_dict))]
        return instance#, {'all_labels': all_labels}

