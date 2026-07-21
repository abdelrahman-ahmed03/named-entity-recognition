<div align="center">

# 🧠 Named Entity Recognition System

### End-to-End NER with LSTM, BiLSTM, BiLSTM-CRF, and DistilBERT

A complete Natural Language Processing project for detecting **people, organizations, locations, and miscellaneous named entities** in English text using traditional deep-learning architectures and a fine-tuned Transformer model.

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-DistilBERT-FFD21E?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CoNLL--2003-6C63FF?style=for-the-badge)

</div>

---

## 📌 Project Overview

This project implements a complete **Named Entity Recognition (NER)** pipeline using the **CoNLL-2003** dataset.

The system recognizes four entity categories:

| Code | Entity Type | Example |
|:---:|---|---|
| `PER` | Person | Elon Musk |
| `ORG` | Organization | Microsoft |
| `LOC` | Location | Washington |
| `MISC` | Miscellaneous | European Union |

The project covers the full machine-learning lifecycle:

- Exploratory data analysis
- Text and label preprocessing
- Word and character vocabulary construction
- GloVe embedding integration
- Character-level CNN feature extraction
- LSTM and BiLSTM sequence modeling
- Linear-chain CRF decoding
- DistilBERT fine-tuning
- Entity-level evaluation with Seqeval
- Interactive deployment with Streamlit

---

## ✨ Main Features

- Four trained NER architectures
- Character-level features for handling unseen words
- Pre-trained GloVe word embeddings
- Custom linear-chain CRF implementation
- Fast tokenizer with correct subword-to-word alignment
- Confidence-based entity filtering
- Colored entity highlighting
- Entity positions and confidence scores
- CSV export for extracted entities
- Model-performance comparison dashboard
- Responsive Streamlit interface
- Local Transformer loading without internet access

---

## 🏆 Model Performance

All models were evaluated on the CoNLL-2003 test split using entity-level metrics.

| Model | Precision | Recall | Test F1 |
|---|---:|---:|---:|
| LSTM | 74.07% | 78.90% | 76.41% |
| BiLSTM | 82.84% | 84.19% | 83.51% |
| BiLSTM + CRF | 84.47% | 84.74% | 84.60% |
| **DistilBERT** | **88.20%** | **89.14%** | **88.67%** |

> **DistilBERT achieved the strongest overall test performance with an F1 score of 88.67%.**

### DistilBERT Performance by Entity Type

| Entity Type | Precision | Recall | F1 |
|---|---:|---:|---:|
| PER | 95.24% | 94.12% | 94.67% |
| LOC | 90.46% | 91.06% | 90.76% |
| ORG | 83.97% | 87.06% | 85.49% |
| MISC | 77.29% | 78.06% | 77.68% |

---

## 📊 Dataset

The project uses the **CoNLL-2003 Named Entity Recognition dataset**.

| Split | Sentences |
|---|---:|
| Training | 14,041 |
| Validation | 3,250 |
| Test | 3,453 |

### IOB2 Label Set

```text
O
B-PER
I-PER
B-ORG
I-ORG
B-LOC
I-LOC
B-MISC
I-MISC
```

### Data Insights

- Training tokens: **203,621**
- Validation tokens: **51,362**
- Test tokens: **46,435**
- Training vocabulary: **23,623 words**
- Maximum sequence length: **128 tokens**
- Invalid IOB transitions: **0**

---

## 🏗️ Model Architectures

### 1. LSTM

```text
Input Tokens
    ↓
GloVe Word Embeddings
    +
Character Embeddings
    ↓
Character CNN
    ↓
Feature Concatenation
    ↓
Unidirectional LSTM
    ↓
Token Classification
```

### 2. BiLSTM

The BiLSTM processes every sentence in both forward and backward directions, allowing each token to use contextual information from both sides.

```text
Word Features + Character Features
                ↓
        Bidirectional LSTM
                ↓
        Token Classification
```

### 3. BiLSTM + CRF

The CRF layer models dependencies between neighboring output tags and improves sequence consistency.

```text
Word + Character Features
            ↓
          BiLSTM
            ↓
      Emission Scores
            ↓
   Linear-Chain CRF Decoder
            ↓
    Optimal Entity Tag Path
```

### 4. DistilBERT

The best-performing model uses contextual Transformer representations and a token-classification head.

```text
Input Text
    ↓
Fast WordPiece Tokenizer
    ↓
Fine-tuned DistilBERT
    ↓
Token Classification Head
    ↓
Subword-to-Word Alignment
    ↓
IOB Entity Reconstruction
```

---

## 🖥️ Streamlit Application

The interactive application allows users to:

- Enter English sentences or paragraphs
- Extract named entities in real time
- Highlight entities using category-specific colors
- View confidence scores
- Adjust the minimum confidence threshold
- Inspect entity start and end positions
- Compare trained model results
- Export detected entities as CSV
- Review dataset and architecture details

### Example

**Input**

```text
Satya Nadella leads Microsoft, which is headquartered in Redmond, Washington.
```

**Detected entities**

| Entity | Type |
|---|---|
| Satya Nadella | PER |
| Microsoft | ORG |
| Redmond | LOC |
| Washington | LOC |

---

## 📁 Project Structure

```text
named_entity_recognition_project/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── processed/
│       ├── word2idx.json
│       ├── char2idx.json
│       ├── label2idx.json
│       └── preprocessing_metadata.json
│
├── models/
│   ├── lstm/
│   │   ├── classification_report.txt
│   │   ├── test_metrics.json
│   │   └── training_history.json
│   │
│   ├── bilstm/
│   │   ├── classification_report.txt
│   │   ├── test_metrics.json
│   │   └── training_history.json
│   │
│   ├── bilstm_crf/
│   │   ├── classification_report.txt
│   │   ├── test_metrics.json
│   │   └── training_history.json
│   │
│   └── transformer/
│       ├── classification_report.txt
│       ├── test_metrics.json
│       ├── training_history.json
│       └── best_model/
│           ├── config.json
│           ├── tokenizer.json
│           └── tokenizer_config.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   └── 03_model_training.ipynb
│
├── reports/
│   └── results/
│       ├── dataset_summary.json
│       ├── entity_distribution.csv
│       ├── oov_statistics.csv
│       ├── split_statistics.csv
│       └── train_tag_distribution.csv
│
└── src/
    ├── __init__.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py
    │   └── preprocess.py
    │
    └── models/
        ├── __init__.py
        ├── lstm_model.py
        ├── bilstm_model.py
        ├── crf.py
        ├── bilstm_crf_model.py
        └── transformer_model.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd named_entity_recognition_project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Linux or macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Model Weights

Large checkpoint files are excluded from Git using `.gitignore`.

The following files remain local and are **not included in the repository**:

```text
models/**/*.pt
models/**/*.pth
models/**/*.bin
models/**/*.safetensors
```

To run the Streamlit application, place the fine-tuned Transformer checkpoint at:

```text
models/transformer/best_model/model.safetensors
```

The complete Transformer directory should contain:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

Traditional checkpoints should be placed in:

```text
models/lstm/
models/bilstm/
models/bilstm_crf/
```

---

## ▶️ Run the Application

From the project root, run:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## 🧪 Local Validation

Compile the project to verify that all Python files are syntactically valid:

```bash
python -m compileall app.py src
```

Test the Transformer model loader:

```bash
python -c "from src.models import load_transformer_ner; tokenizer, model, device = load_transformer_ner('models/transformer/best_model'); print(model.__class__.__name__, tokenizer.is_fast, model.config.num_labels, device)"
```

Expected output includes:

```text
DistilBertForTokenClassification
True
9
cpu
```

---

## 🛠️ Technologies

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Transformers | Hugging Face Transformers, DistilBERT |
| Traditional NLP | GloVe, Character CNN, LSTM, BiLSTM, CRF |
| Evaluation | Seqeval, Scikit-learn |
| Data Processing | Pandas, NumPy, Hugging Face Datasets |
| Interface | Streamlit |
| Visualization | Matplotlib, Streamlit Charts |
| Development | Jupyter Notebook, VS Code, Google Colab |

---

## 📈 Training Environment

- Google Colab
- NVIDIA Tesla T4 GPU
- PyTorch
- Hugging Face Transformers
- Maximum sequence length: 128
- DistilBERT training epochs: 3
- Traditional model batch size: 32
- Transformer batch size: 16

---

## 🔍 Evaluation Strategy

The project uses **entity-level evaluation** rather than token-level accuracy.

Main metrics:

- Precision
- Recall
- Micro F1
- Macro F1
- Weighted F1
- Per-entity classification reports

Entity-level evaluation provides a more meaningful measurement of complete entity detection and boundary accuracy.

---

## 🚀 Future Improvements

- Add multilingual NER support
- Add batch file processing
- Support PDF and document uploads
- Build a REST API using FastAPI
- Deploy the Streamlit application publicly
- Add experiment tracking
- Add model quantization for faster CPU inference
- Train on additional domain-specific datasets
- Add Docker support
- Add automated tests and CI/CD

---

## 👨‍💻 Author

**Abdelrahman Ahmed**

Machine Learning and Natural Language Processing enthusiast focused on building complete AI systems from data preparation to deployment.

---

## ⭐ Support

If this project helped you, consider giving the repository a star.

<div align="center">

**Built with PyTorch, Hugging Face Transformers, and Streamlit**

</div>
