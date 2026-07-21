from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from textwrap import dedent
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

from src.models import get_transformer_label, load_transformer_ner


# =========================================================
# Application configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "transformer" / "best_model"
MAX_LENGTH = 128

ENTITY_CONFIG: dict[str, dict[str, str]] = {
    "PER": {
        "name": "Person",
        "icon": "👤",
        "color": "#ff6b81",
        "soft": "rgba(255, 107, 129, 0.16)",
    },
    "ORG": {
        "name": "Organization",
        "icon": "🏢",
        "color": "#4dabf7",
        "soft": "rgba(77, 171, 247, 0.16)",
    },
    "LOC": {
        "name": "Location",
        "icon": "📍",
        "color": "#51cf66",
        "soft": "rgba(81, 207, 102, 0.16)",
    },
    "MISC": {
        "name": "Miscellaneous",
        "icon": "✨",
        "color": "#ffd43b",
        "soft": "rgba(255, 212, 59, 0.16)",
    },
}

EXAMPLES = {
    "Technology": (
        "Elon Musk founded SpaceX in the United States and later acquired Twitter."
    ),
    "Business": (
        "Satya Nadella leads Microsoft, which is headquartered in Redmond, Washington."
    ),
    "Politics": (
        "Barack Obama met Angela Merkel during a conference in Berlin organized "
        "by the European Union."
    ),
    "Sports": (
        "Lionel Messi joined Inter Miami after playing for Paris Saint-Germain in France."
    ),
    "Research": (
        "Geoffrey Hinton worked at Google and received the Turing Award for his "
        "contributions to artificial intelligence."
    ),
}

MODEL_RESULTS = pd.DataFrame(
    [
        {"Model": "LSTM", "Precision": 74.07, "Recall": 78.90, "Test F1": 76.41},
        {"Model": "BiLSTM", "Precision": 82.84, "Recall": 84.19, "Test F1": 83.51},
        {
            "Model": "BiLSTM + CRF",
            "Precision": 84.47,
            "Recall": 84.74,
            "Test F1": 84.60,
        },
        {
            "Model": "DistilBERT",
            "Precision": 88.20,
            "Recall": 89.14,
            "Test F1": 88.67,
        },
    ]
)

DISTILBERT_ENTITY_RESULTS = pd.DataFrame(
    [
        {"Entity Type": "PER", "Precision": 95.24, "Recall": 94.12, "F1": 94.67},
        {"Entity Type": "LOC", "Precision": 90.46, "Recall": 91.06, "F1": 90.76},
        {"Entity Type": "ORG", "Precision": 83.97, "Recall": 87.06, "F1": 85.49},
        {"Entity Type": "MISC", "Precision": 77.29, "Recall": 78.06, "F1": 77.68},
    ]
)


# =========================================================
# Streamlit page configuration
# =========================================================
st.set_page_config(
    page_title="NER Intelligence Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Session state
# =========================================================
if "ner_input" not in st.session_state:
    st.session_state.ner_input = EXAMPLES["Technology"]

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# =========================================================
# Styling
# =========================================================
def inject_styles() -> None:
    st.markdown(
        dedent(
            """
            <style>
                :root {
                    --page-bg: #090d18;
                    --panel: rgba(21, 28, 47, 0.82);
                    --panel-strong: rgba(25, 34, 57, 0.96);
                    --panel-soft: rgba(255, 255, 255, 0.045);
                    --border: rgba(255, 255, 255, 0.095);
                    --border-strong: rgba(124, 120, 255, 0.28);
                    --text: #f3f6ff;
                    --muted: #9aa6bd;
                    --primary: #7c78ff;
                    --secondary: #20c7ff;
                    --success: #51cf66;
                    --shadow: 0 22px 60px rgba(0, 0, 0, 0.30);
                    --soft-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
                }

                html, body, [class*="css"] {
                    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                        "Segoe UI", sans-serif;
                }

                .stApp {
                    color: var(--text);
                    background:
                        radial-gradient(circle at 10% -5%, rgba(124, 120, 255, 0.18), transparent 31%),
                        radial-gradient(circle at 100% 10%, rgba(32, 199, 255, 0.12), transparent 28%),
                        linear-gradient(180deg, #0b1020 0%, var(--page-bg) 62%, #080b13 100%);
                }

                .block-container {
                    max-width: 1280px;
                    padding-top: 1.35rem;
                    padding-bottom: 3rem;
                }

                header[data-testid="stHeader"] {
                    background: transparent;
                }

                #MainMenu, footer {
                    visibility: hidden;
                }

                h1, h2, h3, h4, h5, h6,
                [data-testid="stMarkdownContainer"] p,
                [data-testid="stMarkdownContainer"] li,
                label {
                    color: var(--text);
                }

                .muted-copy,
                .stCaption,
                [data-testid="stCaptionContainer"] {
                    color: var(--muted) !important;
                }

                /* Sidebar */
                [data-testid="stSidebar"] {
                    background:
                        linear-gradient(180deg, rgba(24, 27, 53, 0.98) 0%, rgba(13, 18, 31, 0.99) 100%);
                    border-right: 1px solid var(--border);
                    box-shadow: 14px 0 35px rgba(0, 0, 0, 0.16);
                }

                [data-testid="stSidebar"] > div:first-child {
                    padding-top: 1.25rem;
                }

                .sidebar-brand {
                    position: relative;
                    overflow: hidden;
                    padding: 1.15rem 1.1rem;
                    margin-bottom: 1.2rem;
                    border: 1px solid rgba(255, 255, 255, 0.11);
                    border-radius: 20px;
                    background:
                        linear-gradient(135deg, rgba(124, 120, 255, 0.30), rgba(32, 199, 255, 0.12));
                    box-shadow: var(--soft-shadow);
                }

                .sidebar-brand::after {
                    content: "";
                    position: absolute;
                    width: 120px;
                    height: 120px;
                    right: -45px;
                    top: -50px;
                    border-radius: 50%;
                    background: rgba(255, 255, 255, 0.08);
                    filter: blur(2px);
                }

                .sidebar-brand-title {
                    position: relative;
                    z-index: 2;
                    font-size: 1.18rem;
                    font-weight: 850;
                    letter-spacing: -0.02em;
                }

                .sidebar-brand-subtitle {
                    position: relative;
                    z-index: 2;
                    margin-top: 0.3rem;
                    color: rgba(239, 243, 255, 0.68);
                    font-size: 0.80rem;
                    line-height: 1.45;
                }

                .model-status {
                    display: flex;
                    align-items: center;
                    gap: 0.65rem;
                    padding: 0.78rem 0.85rem;
                    margin-bottom: 0.95rem;
                    color: #9df1ae;
                    background: rgba(81, 207, 102, 0.09);
                    border: 1px solid rgba(81, 207, 102, 0.24);
                    border-radius: 14px;
                    font-size: 0.84rem;
                    font-weight: 750;
                }

                .status-dot {
                    width: 9px;
                    height: 9px;
                    border-radius: 50%;
                    background: #51cf66;
                    box-shadow: 0 0 0 5px rgba(81, 207, 102, 0.12);
                }

                .legend-card {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.75rem;
                    padding: 0.72rem 0.78rem;
                    margin-bottom: 0.5rem;
                    border-radius: 13px;
                    background: rgba(255, 255, 255, 0.045);
                    border: 1px solid rgba(255, 255, 255, 0.075);
                    transition: transform 0.18s ease, border-color 0.18s ease;
                }

                .legend-card:hover {
                    transform: translateY(-1px);
                    border-color: rgba(124, 120, 255, 0.28);
                }

                .legend-left {
                    display: flex;
                    align-items: center;
                    gap: 0.6rem;
                    font-size: 0.84rem;
                }

                .legend-dot {
                    width: 10px;
                    height: 10px;
                    flex: 0 0 auto;
                    border-radius: 50%;
                    box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.045);
                }

                .legend-code {
                    color: rgba(239, 243, 255, 0.55);
                    font-size: 0.70rem;
                    font-weight: 850;
                    letter-spacing: 0.06em;
                }

                /* Hero */
                .hero-card {
                    position: relative;
                    overflow: hidden;
                    padding: 2.25rem 2.35rem;
                    margin-bottom: 1.35rem;
                    border-radius: 28px;
                    border: 1px solid var(--border-strong);
                    background:
                        linear-gradient(135deg, rgba(124, 120, 255, 0.20), rgba(32, 199, 255, 0.08)),
                        rgba(17, 23, 39, 0.88);
                    box-shadow: var(--shadow);
                    backdrop-filter: blur(16px);
                }

                .hero-card::before {
                    content: "";
                    position: absolute;
                    width: 280px;
                    height: 280px;
                    right: -80px;
                    top: -130px;
                    border-radius: 50%;
                    background: rgba(32, 199, 255, 0.12);
                    filter: blur(3px);
                }

                .hero-card::after {
                    content: "";
                    position: absolute;
                    width: 220px;
                    height: 220px;
                    right: 120px;
                    bottom: -170px;
                    border-radius: 50%;
                    background: rgba(124, 120, 255, 0.13);
                }

                .hero-content {
                    position: relative;
                    z-index: 2;
                }

                .hero-badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.45rem;
                    padding: 0.45rem 0.72rem;
                    margin-bottom: 0.85rem;
                    border-radius: 999px;
                    border: 1px solid rgba(124, 120, 255, 0.31);
                    background: rgba(124, 120, 255, 0.13);
                    color: #c8c6ff;
                    font-size: 0.72rem;
                    font-weight: 850;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                }

                .hero-title {
                    margin: 0;
                    max-width: 900px;
                    font-size: clamp(2.15rem, 4vw, 3.25rem);
                    line-height: 1.03;
                    letter-spacing: -0.045em;
                    font-weight: 900;
                    background: linear-gradient(90deg, #ffffff 0%, #c9c7ff 46%, #65d9ff 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .hero-subtitle {
                    max-width: 850px;
                    margin: 0.95rem 0 0;
                    color: #aab5c9;
                    font-size: 1.00rem;
                    line-height: 1.75;
                }

                .hero-meta {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.6rem;
                    margin-top: 1.25rem;
                }

                .hero-chip {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.4rem;
                    padding: 0.44rem 0.68rem;
                    border-radius: 999px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    background: rgba(255, 255, 255, 0.045);
                    color: #c1cad9;
                    font-size: 0.75rem;
                    font-weight: 700;
                }

                /* Tabs */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 0.55rem;
                    padding: 0.35rem;
                    margin-bottom: 1rem;
                    border: 1px solid var(--border);
                    border-radius: 16px;
                    background: rgba(255, 255, 255, 0.035);
                }

                .stTabs [data-baseweb="tab"] {
                    height: 46px;
                    padding: 0 1rem;
                    border-radius: 12px;
                    color: #aab5c9;
                    font-weight: 750;
                }

                .stTabs [aria-selected="true"] {
                    color: #ffffff !important;
                    background: linear-gradient(90deg, rgba(124, 120, 255, 0.20), rgba(32, 199, 255, 0.11));
                }

                /* Form controls */
                div[data-testid="stTextArea"] textarea {
                    min-height: 175px;
                    padding: 1rem 1.05rem;
                    border-radius: 16px;
                    border: 1px solid rgba(124, 120, 255, 0.25);
                    background: rgba(10, 15, 27, 0.72);
                    color: #f5f7ff;
                    font-size: 0.98rem;
                    line-height: 1.65;
                    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
                }

                div[data-testid="stTextArea"] textarea:focus {
                    border-color: rgba(32, 199, 255, 0.58);
                    box-shadow: 0 0 0 3px rgba(32, 199, 255, 0.10);
                }

                div[data-testid="stButton"] button,
                div[data-testid="stFormSubmitButton"] button,
                div[data-testid="stDownloadButton"] button {
                    min-height: 46px;
                    border-radius: 13px;
                    border: 1px solid rgba(255, 255, 255, 0.09);
                    font-weight: 820;
                    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
                }

                div[data-testid="stButton"] button:hover,
                div[data-testid="stFormSubmitButton"] button:hover,
                div[data-testid="stDownloadButton"] button:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
                    border-color: rgba(124, 120, 255, 0.30);
                }

                div[data-testid="stFormSubmitButton"] button[kind="primary"] {
                    border: 0;
                    color: #ffffff;
                    background: linear-gradient(90deg, #716cff 0%, #24bff1 100%);
                    box-shadow: 0 12px 28px rgba(84, 111, 255, 0.25);
                }

                /* Metrics */
                div[data-testid="stMetric"] {
                    min-height: 108px;
                    padding: 1rem 1.05rem;
                    border: 1px solid var(--border);
                    border-radius: 17px;
                    background:
                        linear-gradient(145deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.025));
                    box-shadow: var(--soft-shadow);
                }

                div[data-testid="stMetricLabel"] {
                    color: #9ba7bc;
                    font-size: 0.78rem;
                    font-weight: 720;
                }

                div[data-testid="stMetricValue"] {
                    color: #f7f8ff;
                    font-weight: 860;
                    letter-spacing: -0.025em;
                }

                [data-testid="stSidebar"] div[data-testid="stMetric"] {
                    min-height: 88px;
                    padding: 0.75rem;
                    border-radius: 14px;
                    box-shadow: none;
                    background: rgba(255, 255, 255, 0.05);
                }

                /* Native containers and dataframes */
                div[data-testid="stVerticalBlockBorderWrapper"] {
                    border-radius: 18px;
                    border-color: var(--border) !important;
                    background: rgba(255, 255, 255, 0.03);
                    box-shadow: var(--soft-shadow);
                }

                div[data-testid="stDataFrame"] {
                    overflow: hidden;
                    border: 1px solid var(--border);
                    border-radius: 16px;
                    box-shadow: var(--soft-shadow);
                }

                /* Custom content cards */
                .section-heading {
                    display: flex;
                    align-items: flex-start;
                    justify-content: space-between;
                    gap: 1rem;
                    margin: 0.25rem 0 1rem;
                }

                .section-heading h3 {
                    margin: 0;
                    font-size: 1.22rem;
                    letter-spacing: -0.02em;
                }

                .section-heading p {
                    margin: 0.3rem 0 0;
                    color: var(--muted);
                    font-size: 0.86rem;
                    line-height: 1.55;
                }

                .empty-state {
                    padding: 2rem 1.5rem;
                    text-align: center;
                    border: 1px dashed rgba(124, 120, 255, 0.28);
                    border-radius: 18px;
                    background: rgba(124, 120, 255, 0.045);
                    color: #a9b4c7;
                }

                .empty-state-icon {
                    display: block;
                    margin-bottom: 0.6rem;
                    font-size: 1.85rem;
                }

                .info-card {
                    height: 100%;
                    padding: 1.2rem 1.25rem;
                    border: 1px solid var(--border);
                    border-radius: 18px;
                    background: rgba(255, 255, 255, 0.035);
                    box-shadow: var(--soft-shadow);
                }

                .info-card h4 {
                    margin: 0 0 0.65rem;
                    font-size: 1rem;
                }

                .info-card p,
                .info-card li {
                    color: #aab5c9;
                    font-size: 0.88rem;
                    line-height: 1.65;
                }

                .info-card ul {
                    padding-left: 1.1rem;
                    margin-bottom: 0;
                }

                .footer-note {
                    margin-top: 2.4rem;
                    padding-top: 1rem;
                    border-top: 1px solid rgba(255, 255, 255, 0.07);
                    text-align: center;
                    color: rgba(171, 182, 203, 0.56);
                    font-size: 0.76rem;
                    letter-spacing: 0.01em;
                }

                @media (max-width: 900px) {
                    .block-container {
                        padding-left: 1rem;
                        padding-right: 1rem;
                    }

                    .hero-card {
                        padding: 1.65rem 1.4rem;
                        border-radius: 22px;
                    }

                    .hero-subtitle {
                        font-size: 0.94rem;
                    }
                }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


inject_styles()


# =========================================================
# Model loading
# =========================================================
@st.cache_resource(show_spinner=False)
def load_ner_model():
    return load_transformer_ner(
        model_path=MODEL_PATH,
        local_files_only=True,
    )


# =========================================================
# Inference helpers
# =========================================================
def extract_entities(
    text: str,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    full_encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )
    was_truncated = len(full_encoding["input_ids"]) > MAX_LENGTH

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    offsets = encoded["offset_mapping"][0].tolist()
    word_ids = encoded.word_ids(batch_index=0)

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }

    with torch.inference_mode():
        logits = model(**model_inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1).cpu()
        predicted_ids = probabilities.argmax(dim=-1).tolist()

    words: list[dict[str, Any]] = []
    processed_word_ids: set[int] = set()

    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in processed_word_ids:
            continue

        processed_word_ids.add(word_id)

        start = offsets[token_index][0]
        end = offsets[token_index][1]
        next_index = token_index + 1

        while next_index < len(word_ids) and word_ids[next_index] == word_id:
            end = offsets[next_index][1]
            next_index += 1

        label_id = int(predicted_ids[token_index])
        label = get_transformer_label(model, label_id)
        confidence = float(probabilities[token_index][label_id])

        words.append(
            {
                "text": text[start:end],
                "start": start,
                "end": end,
                "label": label,
                "confidence": confidence,
            }
        )

    entities: list[dict[str, Any]] = []
    current_entity: dict[str, Any] | None = None

    for word in words:
        label = word["label"]

        if label == "O" or "-" not in label:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        prefix, entity_type = label.split("-", 1)
        should_start_new = (
            prefix == "B"
            or current_entity is None
            or current_entity["type"] != entity_type
        )

        if should_start_new:
            if current_entity is not None:
                entities.append(current_entity)

            current_entity = {
                "text": word["text"],
                "type": entity_type,
                "start": word["start"],
                "end": word["end"],
                "scores": [word["confidence"]],
            }
        else:
            current_entity["end"] = word["end"]
            current_entity["text"] = text[
                current_entity["start"] : current_entity["end"]
            ]
            current_entity["scores"].append(word["confidence"])

    if current_entity is not None:
        entities.append(current_entity)

    for entity in entities:
        scores = entity.pop("scores")
        entity["confidence"] = sum(scores) / len(scores)

    return {
        "text": text,
        "entities": entities,
        "word_count": len(words),
        "token_count": int(encoded["attention_mask"][0].sum().item()),
        "was_truncated": was_truncated,
    }


def build_highlighted_html(text: str, entities: list[dict[str, Any]]) -> str:
    html_parts = [
        dedent(
            """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8" />
                <style>
                    * { box-sizing: border-box; }
                    body {
                        margin: 0;
                        padding: 0;
                        background: transparent;
                        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    }
                    .result-shell {
                        position: relative;
                        overflow: hidden;
                        padding: 24px 25px;
                        min-height: 122px;
                        border-radius: 18px;
                        border: 1px solid rgba(255, 255, 255, 0.10);
                        background:
                            radial-gradient(circle at 95% 0%, rgba(32, 199, 255, 0.08), transparent 30%),
                            linear-gradient(145deg, rgba(26, 34, 56, 0.98), rgba(14, 19, 33, 0.98));
                        box-shadow: 0 18px 38px rgba(0, 0, 0, 0.26);
                        color: #f2f5fb;
                        font-size: 17px;
                        line-height: 2.55;
                        white-space: pre-wrap;
                        overflow-wrap: anywhere;
                    }
                    .entity {
                        display: inline-flex;
                        align-items: center;
                        gap: 7px;
                        padding: 4px 8px 4px 9px;
                        margin: 2px 3px;
                        border: 1px solid rgba(255, 255, 255, 0.16);
                        border-radius: 10px;
                        color: #111827;
                        font-weight: 780;
                        line-height: 1.65;
                        box-shadow: 0 5px 14px rgba(0, 0, 0, 0.20);
                    }
                    .entity-label {
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        min-width: 31px;
                        padding: 2px 5px;
                        border-radius: 6px;
                        background: rgba(255, 255, 255, 0.42);
                        font-size: 9px;
                        font-weight: 950;
                        letter-spacing: 0.06em;
                    }
                </style>
            </head>
            <body>
                <div class="result-shell">
            """
        )
    ]

    previous_end = 0

    for entity in entities:
        html_parts.append(escape(text[previous_end : entity["start"]]))

        entity_type = entity["type"]
        color = ENTITY_CONFIG.get(entity_type, {"color": "#ced4da"})["color"]
        confidence = entity["confidence"] * 100

        html_parts.append(
            f'<span class="entity" style="background:{color};" '
            f'title="Confidence: {confidence:.2f}%">'
            f'{escape(entity["text"])}'
            f'<span class="entity-label">{escape(entity_type)}</span>'
            "</span>"
        )
        previous_end = entity["end"]

    html_parts.append(escape(text[previous_end:]))
    html_parts.append("</div></body></html>")
    return "".join(html_parts)


def build_entity_dataframe(entities: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for index, entity in enumerate(entities, start=1):
        entity_type = entity["type"]
        rows.append(
            {
                "#": index,
                "Entity": entity["text"],
                "Type": entity_type,
                "Category": ENTITY_CONFIG.get(
                    entity_type, {"name": entity_type}
                )["name"],
                "Confidence": round(entity["confidence"] * 100, 2),
                "Start": entity["start"],
                "End": entity["end"],
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.markdown(
        dedent(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">🧠 NER Intelligence</div>
                <div class="sidebar-brand-subtitle">
                    Professional named-entity extraction powered by a fine-tuned transformer.
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Active model")
    st.markdown(
        dedent(
            """
            <div class="model-status">
                <span class="status-dot"></span>
                DistilBERT model ready
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.metric("Test F1", "88.67%")
        st.metric("Precision", "88.20%")
    with metric_right:
        st.metric("Validation F1", "93.08%")
        st.metric("Recall", "89.14%")

    st.markdown("### Entity legend")
    for entity_code, entity_data in ENTITY_CONFIG.items():
        st.markdown(
            (
                '<div class="legend-card">'
                '<div class="legend-left">'
                f'<span class="legend-dot" style="background:{entity_data["color"]};"></span>'
                f'<span>{entity_data["icon"]} {entity_data["name"]}</span>'
                "</div>"
                f'<span class="legend-code">{entity_code}</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### Detection settings")
    confidence_threshold = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Entities below this confidence level are hidden from the results.",
    )

    selected_example = st.selectbox(
        "Quick example",
        options=list(EXAMPLES.keys()),
    )

    action_left, action_right = st.columns(2)
    with action_left:
        if st.button("Load", use_container_width=True):
            st.session_state.ner_input = EXAMPLES[selected_example]
            st.session_state.analysis_result = None
            st.rerun()

    with action_right:
        if st.button("Clear", use_container_width=True):
            st.session_state.ner_input = ""
            st.session_state.analysis_result = None
            st.rerun()

    st.divider()
    st.caption(
        "Fast tokenizer → DistilBERT → token classification → IOB reconstruction"
    )


# =========================================================
# Load model
# =========================================================
try:
    with st.spinner("Loading the fine-tuned DistilBERT model..."):
        tokenizer, model, device = load_ner_model()
except Exception as error:
    st.error("The model could not be loaded.")
    st.code(str(error))
    st.stop()


# =========================================================
# Header
# =========================================================
st.markdown(
    dedent(
        f"""
        <div class="hero-card">
            <div class="hero-content">
                <div class="hero-badge">✦ Natural Language Processing</div>
                <h1 class="hero-title">Named Entity Recognition Studio</h1>
                <p class="hero-subtitle">
                    Detect people, organizations, locations, and miscellaneous entities in real time
                    using a fine-tuned DistilBERT model trained on the CoNLL-2003 dataset.
                </p>
                <div class="hero-meta">
                    <span class="hero-chip">🤖 DistilBERT</span>
                    <span class="hero-chip">🏷️ 9 IOB labels</span>
                    <span class="hero-chip">📏 Max length {MAX_LENGTH}</span>
                    <span class="hero-chip">💻 {escape(str(device).upper())}</span>
                </div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# =========================================================
# Application tabs
# =========================================================
extraction_tab, performance_tab, about_tab = st.tabs(
    [
        "🔎 Entity Extraction",
        "📊 Model Performance",
        "ℹ️ Project Overview",
    ]
)


# =========================================================
# Entity extraction tab
# =========================================================
with extraction_tab:
    st.markdown(
        dedent(
            """
            <div class="section-heading">
                <div>
                    <h3>Analyze English text</h3>
                    <p>Enter a sentence or paragraph, then extract and inspect its named entities.</p>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    with st.form("ner_analysis_form"):
        input_text = st.text_area(
            "Input text",
            key="ner_input",
            height=185,
            max_chars=4000,
            placeholder="Example: Tim Cook works at Apple in California.",
        )
        analyze_button = st.form_submit_button(
            "✨ Extract Named Entities",
            type="primary",
            use_container_width=True,
        )

    if analyze_button:
        cleaned_text = input_text.strip()

        if not cleaned_text:
            st.warning("Please enter English text before starting the analysis.")
        else:
            started_at = perf_counter()
            with st.spinner("Analyzing text..."):
                result = extract_entities(
                    text=cleaned_text,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )
            result["elapsed_ms"] = (perf_counter() - started_at) * 1000
            st.session_state.analysis_result = result

    analysis_result = st.session_state.analysis_result

    if analysis_result is None:
        st.markdown(
            dedent(
                """
                <div class="empty-state">
                    <span class="empty-state-icon">⌁</span>
                    Enter text and run the model to view highlighted entities, confidence scores,
                    detailed offsets, and downloadable results.
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    else:
        filtered_entities = [
            entity
            for entity in analysis_result["entities"]
            if entity["confidence"] >= confidence_threshold
        ]
        entity_counts = Counter(entity["type"] for entity in filtered_entities)

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("🔎 Entities found", len(filtered_entities))
        with metric_col2:
            st.metric("🧾 Input characters", f'{len(analysis_result["text"]):,}')
        with metric_col3:
            st.metric("⚡ Inference time", f'{analysis_result["elapsed_ms"]:.0f} ms')
        with metric_col4:
            st.metric("💻 Runtime device", str(device).upper())

        if analysis_result["was_truncated"]:
            st.warning(
                f"The input exceeded the model limit. Only the first {MAX_LENGTH} tokens were analyzed."
            )

        st.markdown("### Highlighted text")
        result_height = max(
            190,
            min(500, 165 + len(analysis_result["text"]) // 3),
        )
        components.html(
            build_highlighted_html(
                analysis_result["text"],
                filtered_entities,
            ),
            height=result_height,
            scrolling=False,
        )

        if not filtered_entities:
            st.info(
                "No named entities passed the selected confidence threshold. "
                "Try lowering the threshold in the sidebar."
            )
        else:
            st.markdown("### Detected entity details")
            entity_dataframe = build_entity_dataframe(filtered_entities)

            st.dataframe(
                entity_dataframe,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Model confidence for the detected entity.",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%",
                    )
                },
            )

            breakdown_columns = st.columns(4)
            for column, entity_type in zip(
                breakdown_columns,
                ["PER", "ORG", "LOC", "MISC"],
            ):
                entity_data = ENTITY_CONFIG[entity_type]
                with column:
                    st.metric(
                        f'{entity_data["icon"]} {entity_data["name"]}',
                        entity_counts.get(entity_type, 0),
                    )

            csv_data = entity_dataframe.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download detected entities as CSV",
                data=csv_data,
                file_name="detected_entities.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.success(
                f"Analysis completed successfully. {len(filtered_entities)} entities are currently visible."
            )


# =========================================================
# Performance tab
# =========================================================
with performance_tab:
    st.markdown(
        dedent(
            """
            <div class="section-heading">
                <div>
                    <h3>Model architecture comparison</h3>
                    <p>Entity-level seqeval metrics measured on the CoNLL-2003 test set.</p>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    score_col1, score_col2, score_col3 = st.columns(3)
    with score_col1:
        st.metric("Best model", "DistilBERT")
    with score_col2:
        st.metric("Best test F1", "88.67%")
    with score_col3:
        st.metric("Improvement over LSTM", "+12.26 pts")

    formatted_results = MODEL_RESULTS.copy()
    for column in ["Precision", "Recall", "Test F1"]:
        formatted_results[column] = formatted_results[column].map(lambda value: f"{value:.2f}%")

    st.dataframe(
        formatted_results,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Test F1 comparison")
    st.bar_chart(
        MODEL_RESULTS[["Model", "Test F1"]].set_index("Model"),
        height=380,
    )

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        with st.container(border=True):
            st.markdown("#### Best traditional architecture")
            st.markdown("## BiLSTM + CRF")
            st.metric("Test F1", "84.60%")
            st.caption(
                "The CRF decoder improved sequence consistency and entity-boundary decisions."
            )

    with result_col2:
        with st.container(border=True):
            st.markdown("#### Overall best architecture")
            st.markdown("## DistilBERT")
            st.metric("Test F1", "88.67%")
            st.caption(
                "Contextual subword representations produced the strongest overall results."
            )

    st.markdown("### DistilBERT performance by entity type")
    entity_results = DISTILBERT_ENTITY_RESULTS.copy()
    for column in ["Precision", "Recall", "F1"]:
        entity_results[column] = entity_results[column].map(lambda value: f"{value:.2f}%")

    st.dataframe(
        entity_results,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# Project overview tab
# =========================================================
with about_tab:
    overview_col1, overview_col2 = st.columns(2)

    with overview_col1:
        st.markdown(
            dedent(
                """
                <div class="info-card">
                    <h4>📚 Dataset</h4>
                    <ul>
                        <li><strong>Dataset:</strong> CoNLL-2003</li>
                        <li><strong>Train sentences:</strong> 14,041</li>
                        <li><strong>Validation sentences:</strong> 3,250</li>
                        <li><strong>Test sentences:</strong> 3,453</li>
                        <li><strong>Tagging scheme:</strong> IOB2</li>
                        <li><strong>Entity groups:</strong> PER, ORG, LOC, MISC</li>
                    </ul>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        st.markdown(
            dedent(
                """
                <div class="info-card">
                    <h4>🧩 Traditional architectures</h4>
                    <ul>
                        <li>Pre-trained GloVe word embeddings</li>
                        <li>Character-level CNN representations</li>
                        <li>Unidirectional LSTM baseline</li>
                        <li>Bidirectional LSTM encoder</li>
                        <li>Linear-chain CRF sequence decoder</li>
                    </ul>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with overview_col2:
        st.markdown(
            dedent(
                """
                <div class="info-card">
                    <h4>🤖 Transformer pipeline</h4>
                    <ul>
                        <li>Cased fast WordPiece tokenizer</li>
                        <li>Fine-tuned DistilBERT encoder</li>
                        <li>Token-classification output head</li>
                        <li>First-subword prediction alignment</li>
                        <li>IOB reconstruction into complete entities</li>
                    </ul>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        st.markdown(
            dedent(
                """
                <div class="info-card">
                    <h4>📏 Evaluation and delivery</h4>
                    <ul>
                        <li>Entity-level precision, recall, and F1</li>
                        <li>Per-entity classification reports</li>
                        <li>Model architecture comparison</li>
                        <li>Local inference with CPU or CUDA</li>
                        <li>Interactive Streamlit application</li>
                    </ul>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### End-to-end workflow")
    workflow_col1, workflow_col2, workflow_col3, workflow_col4 = st.columns(4)
    workflow_items = [
        (workflow_col1, "1", "Prepare", "Clean and encode CoNLL-2003 sequences."),
        (workflow_col2, "2", "Train", "Compare recurrent and transformer models."),
        (workflow_col3, "3", "Evaluate", "Measure entity-level seqeval metrics."),
        (workflow_col4, "4", "Deploy", "Serve the best model through Streamlit."),
    ]

    for column, number, title, description in workflow_items:
        with column:
            with st.container(border=True):
                st.markdown(f"### {number}")
                st.markdown(f"**{title}**")
                st.caption(description)


# =========================================================
# Footer
# =========================================================
st.markdown(
    (
        '<div class="footer-note">'
        "Named Entity Recognition System · CoNLL-2003 · PyTorch · "
        "Hugging Face Transformers · Streamlit"
        "</div>"
    ),
    unsafe_allow_html=True,
)
