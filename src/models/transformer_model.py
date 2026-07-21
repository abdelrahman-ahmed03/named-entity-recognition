from __future__ import annotations

from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_transformer_ner(
    model_path: str | Path,
    device: torch.device | None = None,
    local_files_only: bool = True,
) -> tuple[
    PreTrainedTokenizerBase,
    PreTrainedModel,
    torch.device,
]:
    """
    Load the fine-tuned transformer NER model and tokenizer.

    Parameters
    ----------
    model_path:
        Path to the saved Hugging Face model directory.

    device:
        Optional PyTorch device. If omitted, CUDA is used when
        available; otherwise CPU is used.

    local_files_only:
        Prevent downloading files from the internet when True.
    """

    model_directory = Path(model_path)

    if not model_directory.exists():
        raise FileNotFoundError(
            f"Transformer model directory not found: "
            f"{model_directory}"
        )

    if not model_directory.is_dir():
        raise NotADirectoryError(
            f"Expected a model directory, received: "
            f"{model_directory}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        use_fast=True,
        local_files_only=local_files_only,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for word alignment."
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_directory,
        local_files_only=local_files_only,
    )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def get_transformer_label(
    model: PreTrainedModel,
    label_id: int,
) -> str:
    """
    Convert a numeric prediction ID to its NER label.
    """

    id_to_label = model.config.id2label

    if label_id in id_to_label:
        return id_to_label[label_id]

    if str(label_id) in id_to_label:
        return id_to_label[str(label_id)]

    return str(label_id)