import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)


class CharacterCNN(nn.Module):
    """
    Extract character-level features for every token.
    """

    def __init__(
        self,
        character_vocabulary_size,
        character_embedding_dim=30,
        character_feature_dim=50,
        padding_idx=0
    ):
        super().__init__()

        self.character_embedding = nn.Embedding(
            num_embeddings=character_vocabulary_size,
            embedding_dim=character_embedding_dim,
            padding_idx=padding_idx
        )

        self.character_convolution = nn.Conv1d(
            in_channels=character_embedding_dim,
            out_channels=character_feature_dim,
            kernel_size=3,
            padding=1
        )

        self.activation = nn.ReLU()

    def forward(self, character_ids):
        """
        character_ids shape:
        [batch_size, sequence_length, max_word_length]
        """

        batch_size, sequence_length, max_word_length = (
            character_ids.shape
        )

        flattened_characters = character_ids.reshape(
            batch_size * sequence_length,
            max_word_length
        )

        embedded_characters = self.character_embedding(
            flattened_characters
        )

        # [batch*sequence, char_embedding, word_length]
        embedded_characters = embedded_characters.transpose(1, 2)

        character_features = self.character_convolution(
            embedded_characters
        )

        character_features = self.activation(character_features)

        # Max pooling over characters
        character_features, _ = torch.max(
            character_features,
            dim=2
        )

        character_features = character_features.reshape(
            batch_size,
            sequence_length,
            -1
        )

        return character_features


class LSTMNERModel(nn.Module):
    """
    Word Embedding + Character CNN + Unidirectional LSTM.
    """

    def __init__(
        self,
        word_vocabulary_size,
        character_vocabulary_size,
        number_of_labels,
        word_embedding_dim=100,
        character_embedding_dim=30,
        character_feature_dim=50,
        lstm_hidden_dim=128,
        lstm_layers=1,
        dropout=0.3,
        word_padding_idx=0,
        character_padding_idx=0
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(
            num_embeddings=word_vocabulary_size,
            embedding_dim=word_embedding_dim,
            padding_idx=word_padding_idx
        )

        self.character_encoder = CharacterCNN(
            character_vocabulary_size=character_vocabulary_size,
            character_embedding_dim=character_embedding_dim,
            character_feature_dim=character_feature_dim,
            padding_idx=character_padding_idx
        )

        combined_embedding_dim = (
            word_embedding_dim + character_feature_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=combined_embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.output_dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            lstm_hidden_dim,
            number_of_labels
        )

    def forward(
        self,
        word_ids,
        character_ids,
        sequence_lengths
    ):
        """
        Returns:
            logits shape:
            [batch_size, sequence_length, number_of_labels]
        """

        word_features = self.word_embedding(word_ids)

        character_features = self.character_encoder(
            character_ids
        )

        combined_features = torch.cat(
            [word_features, character_features],
            dim=-1
        )

        combined_features = self.embedding_dropout(
            combined_features
        )

        packed_features = pack_padded_sequence(
            combined_features,
            lengths=sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_features)

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=word_ids.size(1)
        )

        lstm_output = self.output_dropout(lstm_output)

        logits = self.classifier(lstm_output)

        return logits