from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from .lstm_model import CharacterCNN


class BiLSTMBackbone(nn.Module):
    """
    Word embeddings + Character CNN + Bidirectional LSTM.
    """

    def __init__(
        self,
        word_vocabulary_size: int,
        character_vocabulary_size: int,
        word_embedding_dim: int = 100,
        character_embedding_dim: int = 30,
        character_feature_dim: int = 50,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        word_padding_idx: int = 0,
        character_padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self.word_embedding = nn.Embedding(
            num_embeddings=word_vocabulary_size,
            embedding_dim=word_embedding_dim,
            padding_idx=word_padding_idx,
        )

        self.character_encoder = CharacterCNN(
            character_vocabulary_size=character_vocabulary_size,
            character_embedding_dim=character_embedding_dim,
            character_feature_dim=character_feature_dim,
            padding_idx=character_padding_idx,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=word_embedding_dim + character_feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(
        self,
        word_ids: torch.Tensor,
        character_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        word_features = self.word_embedding(word_ids)

        character_features = self.character_encoder(
            character_ids
        )

        combined_features = torch.cat(
            [word_features, character_features],
            dim=-1,
        )

        combined_features = self.embedding_dropout(
            combined_features
        )

        packed_features = pack_padded_sequence(
            combined_features,
            lengths=sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.bilstm(
            packed_features
        )

        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=word_ids.size(1),
        )

        return self.output_dropout(output)


class BiLSTMSoftmaxTagger(nn.Module):
    """
    BiLSTM backbone followed by a token-level classifier.
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

        self.backbone = BiLSTMBackbone(
            word_vocabulary_size=word_vocabulary_size,
            character_vocabulary_size=character_vocabulary_size,
            word_embedding_dim=word_embedding_dim,
            character_embedding_dim=character_embedding_dim,
            character_feature_dim=character_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            word_padding_idx=word_padding_idx,
            character_padding_idx=character_padding_idx,
        )

        self.classifier = nn.Linear(
            in_features=self.backbone.output_dim,
            out_features=number_of_labels,
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

        return self.classifier(features)