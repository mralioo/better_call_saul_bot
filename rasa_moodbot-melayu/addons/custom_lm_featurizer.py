import logging
from typing import List, Text, Tuple

import numpy as np
import transformers
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import \
    LanguageModelFeaturizer
from rasa.nlu.utils.hugging_face.registry import (
    model_embeddings_post_processors, model_special_tokens_pre_processors,
    model_tokens_cleaners)
from transformers import BertTokenizer, TFBertForMaskedLM

logger = logging.getLogger(__name__)
logger.info(f"Using transformers library version {transformers.__version__}")


def melayubert_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add MelayuBERT style special tokens(CLS and SEP).

    Args:
        token_ids: List of token ids without any special tokens.

    Returns:
        List of token ids augmented with special tokens.
    """
    # see the model's vocab.txt to get the right index
    BERT_CLS_ID = 3
    BERT_SEP_ID = 1

    processed_tokens = token_ids

    processed_tokens.insert(0, BERT_CLS_ID)
    processed_tokens.append(BERT_SEP_ID)

    return processed_tokens


class CustomLanguageModelFeaturizer(LanguageModelFeaturizer):
    def _load_model_metadata(self) -> None:
        self.model_name = self.component_config["model_name"]
        self.model_weights = self.component_config["model_weights"]
        self.cache_dir = self.component_config["cache_dir"]
        self.max_model_sequence_length = 512

        if self.model_name != "StevenLimcorn/MelayuBERT":
            raise KeyError(f"'{self.model_name}' is not a supported model!")

        if not self.model_weights:
            raise RuntimeError(
                f"No model weights found in '{self.model_weights}'. Please download model weights from https://huggingface.co/StevenLimcorn/MelayuBERT"
            )

    def _load_model_instance(self, skip_model_load: bool) -> None:
        logger.debug(f"Loading Tokenizer and Model for {self.model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_weights)
        self.model = TFBertForMaskedLM.from_pretrained(
            self.model_weights, output_hidden_states=True
        )
        self.pad_token_id = self.tokenizer.unk_token_id

    def _lm_specific_token_cleanup(
        self, split_token_ids: List[int], token_strings: List[Text]
    ) -> Tuple[List[int], List[Text]]:
        return model_tokens_cleaners["bert"](split_token_ids, token_strings)

    def _add_lm_specific_special_tokens(
        self, token_ids: List[List[int]]
    ) -> List[List[int]]:
        augmented_tokens = [
            melayubert_tokens_pre_processor(example_token_ids)
            for example_token_ids in token_ids
        ]
        return augmented_tokens

    def _post_process_sequence_embeddings(
        self, sequence_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        sentence_embeddings = []
        post_processed_sequence_embeddings = []

        for example_embedding in sequence_embeddings:
            (
                example_sentence_embedding,
                example_post_processed_embedding,
            ) = model_embeddings_post_processors["bert"](example_embedding)

            sentence_embeddings.append(example_sentence_embedding)
            post_processed_sequence_embeddings.append(example_post_processed_embedding)

        return (
            np.array(sentence_embeddings),
            np.array(post_processed_sequence_embeddings),
        )

    def _compute_batch_sequence_features(
        self, batch_attention_mask: np.ndarray, padded_token_ids: List[List[int]]
    ) -> np.ndarray:
        model_outputs = self.model(
            np.array(padded_token_ids), attention_mask=np.array(batch_attention_mask)
        )
        sequence_hidden_states = model_outputs.hidden_states[-1]

        sequence_hidden_states = sequence_hidden_states.numpy()

        return sequence_hidden_states
