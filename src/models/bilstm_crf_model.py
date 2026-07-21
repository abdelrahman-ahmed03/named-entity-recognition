from __future__ import annotations

import torch
import torch.nn as nn

from .bilstm_model import BiLSTMBackbone
from .crf import LinearChainCRF


class BiLSTMCRFTagger(nn.Module):
    """
    Character CNN + word embeddings + BiLSTM +
    linear-chain CRF.
    """

    def __init__(
        self,
        word_vocabulary_size: int,
        character_vocabulary_size: int,
        number_of_labels: int,
        word_embedding_dim: int = 100,
        character_embedding_dim: int = 30,
        character_feature_dim: int = 50,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        word_padding_idx: int = 0,
        character_padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self.number_of_labels = number_of_labels

        self.backbone = BiLSTMBackbone(
            word_vocabulary_size=word_vocabulary_size,
            character_vocabulary_size=(
                character_vocabulary_size
            ),
            word_embedding_dim=word_embedding_dim,
            character_embedding_dim=(
                character_embedding_dim
            ),
            character_feature_dim=(
                character_feature_dim
            ),
            hidden_dim=hidden_dim,
            dropout=dropout,
            word_padding_idx=word_padding_idx,
            character_padding_idx=(
                character_padding_idx
            ),
        )

        self.emission_layer = nn.Linear(
            in_features=self.backbone.output_dim,
            out_features=number_of_labels,
        )

        self.crf = LinearChainCRF(
            number_of_tags=number_of_labels
        )

    def forward(
        self,
        word_ids: torch.Tensor,
        character_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(
            word_ids=word_ids,
            character_ids=character_ids,
            sequence_lengths=sequence_lengths,
        )

        return self.emission_layer(features)

    def calculate_loss(
        self,
        word_ids: torch.Tensor,
        character_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        label_padding_id: int = -100,
    ) -> torch.Tensor:
        emissions = self.forward(
            word_ids=word_ids,
            character_ids=character_ids,
            sequence_lengths=sequence_lengths,
        )

        mask = attention_mask.bool()

        safe_labels = labels.masked_fill(
            ~mask,
            0,
        )

        safe_labels = safe_labels.masked_fill(
            safe_labels == label_padding_id,
            0,
        )

        return self.crf.negative_log_likelihood(
            emissions=emissions,
            tags=safe_labels,
            mask=mask,
        )

    def decode(
        self,
        word_ids: torch.Tensor,
        character_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        emissions = self.forward(
            word_ids=word_ids,
            character_ids=character_ids,
            sequence_lengths=sequence_lengths,
        )

        return self.crf.decode(
            emissions=emissions,
            mask=attention_mask.bool(),
        )