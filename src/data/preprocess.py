import json
from collections import Counter
from pathlib import Path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"


def build_word_vocabulary(tokens_column):
    """
    Build word vocabulary from training tokens only.
    """

    word_counter = Counter(
        token
        for sentence in tokens_column
        for token in sentence
    )

    word2idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    sorted_words = sorted(
        word_counter.items(),
        key=lambda item: (-item[1], item[0])
    )

    for word, _ in sorted_words:
        word2idx[word] = len(word2idx)

    return word2idx


def build_character_vocabulary(tokens_column):
    """
    Build character vocabulary from training tokens only.
    """

    character_counter = Counter(
        character
        for sentence in tokens_column
        for token in sentence
        for character in token
    )

    char2idx = {
        PAD_CHAR: 0,
        UNK_CHAR: 1
    }

    sorted_characters = sorted(
        character_counter.items(),
        key=lambda item: (-item[1], item[0])
    )

    for character, _ in sorted_characters:
        char2idx[character] = len(char2idx)

    return char2idx


def save_json(data, output_path):
    """
    Save a Python dictionary to JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def load_json(file_path):
    """
    Load a JSON file.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)