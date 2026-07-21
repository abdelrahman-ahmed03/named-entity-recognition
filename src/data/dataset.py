import numpy as np
import torch
from torch.utils.data import Dataset


PAD_WORD_ID = 0
UNK_WORD_ID = 1

PAD_CHAR_ID = 0
UNK_CHAR_ID = 1

LABEL_PAD_ID = -100


def encode_sample(
    tokens,
    label_ids,
    word2idx,
    char2idx,
    max_sequence_length=128,
    max_word_length=16
):
    """
    Convert one NER sample into fixed-size numerical arrays.
    """

    if len(tokens) != len(label_ids):
        raise ValueError(
            "Number of tokens must equal number of labels."
        )

    tokens = tokens[:max_sequence_length]
    label_ids = label_ids[:max_sequence_length]

    sequence_length = len(tokens)
    padding_length = max_sequence_length - sequence_length

    word_ids = [
        word2idx.get(token, UNK_WORD_ID)
        for token in tokens
    ]

    character_ids = []

    for token in tokens:
        token_character_ids = [
            char2idx.get(character, UNK_CHAR_ID)
            for character in token[:max_word_length]
        ]

        character_padding_length = (
            max_word_length - len(token_character_ids)
        )

        token_character_ids += (
            [PAD_CHAR_ID] * character_padding_length
        )

        character_ids.append(token_character_ids)

    attention_mask = [1] * sequence_length

    word_ids += [PAD_WORD_ID] * padding_length
    label_ids = list(label_ids) + [LABEL_PAD_ID] * padding_length
    attention_mask += [0] * padding_length

    character_ids += [
        [PAD_CHAR_ID] * max_word_length
        for _ in range(padding_length)
    ]

    return {
        "word_ids": np.asarray(
            word_ids,
            dtype=np.int64
        ),
        "character_ids": np.asarray(
            character_ids,
            dtype=np.int64
        ),
        "labels": np.asarray(
            label_ids,
            dtype=np.int64
        ),
        "attention_mask": np.asarray(
            attention_mask,
            dtype=np.bool_
        ),
        "sequence_length": sequence_length
    }


class NERDataset(Dataset):
    """
    PyTorch Dataset for NER samples.
    """

    def __init__(
        self,
        dataset_split,
        word2idx,
        char2idx,
        max_sequence_length=128,
        max_word_length=16
    ):
        self.dataset_split = dataset_split
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.max_sequence_length = max_sequence_length
        self.max_word_length = max_word_length

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, index):
        sample = self.dataset_split[index]

        encoded_sample = encode_sample(
            tokens=sample["tokens"],
            label_ids=sample["ner_tags"],
            word2idx=self.word2idx,
            char2idx=self.char2idx,
            max_sequence_length=self.max_sequence_length,
            max_word_length=self.max_word_length
        )

        return {
            "word_ids": torch.tensor(
                encoded_sample["word_ids"],
                dtype=torch.long
            ),
            "character_ids": torch.tensor(
                encoded_sample["character_ids"],
                dtype=torch.long
            ),
            "labels": torch.tensor(
                encoded_sample["labels"],
                dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                encoded_sample["attention_mask"],
                dtype=torch.bool
            ),
            "sequence_length": torch.tensor(
                encoded_sample["sequence_length"],
                dtype=torch.long
            )
        }
    