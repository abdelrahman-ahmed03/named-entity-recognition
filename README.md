<div align="center">

# 🧠 Named Entity Recognition System

### End-to-End NER with LSTM, BiLSTM, BiLSTM-CRF, and DistilBERT

A complete Natural Language Processing system for detecting **people, organizations, locations, and miscellaneous named entities** in English text using traditional deep-learning architectures and a fine-tuned Transformer model.

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-DistilBERT-FFD21E?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CoNLL--2003-6C63FF?style=for-the-badge)

<br>

[Overview](#-project-overview) •
[Preview](#-application-preview) •
[Performance](#-model-performance) •
[Installation](#️-installation) •
[Author](#-author)

</div>

---

## 📌 Project Overview

This project implements a complete **Named Entity Recognition (NER)** pipeline using the **CoNLL-2003** dataset.

The system identifies four named-entity categories:

| Code | Entity Type | Example |
|:---:|---|---|
| `PER` | Person | Geoffrey Hinton |
| `ORG` | Organization | Google |
| `LOC` | Location | Washington |
| `MISC` | Miscellaneous | Turing Award |

The project covers the complete machine-learning lifecycle:

- Exploratory data analysis
- Text and label preprocessing
- Word and character vocabulary construction
- GloVe embedding integration
- Character-level CNN feature extraction
- LSTM and BiLSTM sequence modeling
- Linear-chain CRF decoding
- DistilBERT fine-tuning
- Entity-level evaluation using Seqeval
- Interactive deployment using Streamlit

---

## ✨ Main Features

- Four trained Named Entity Recognition architectures
- Character-level features for handling unseen words
- Pre-trained GloVe word embeddings
- Custom linear-chain CRF implementation
- Fine-tuned DistilBERT model
- Correct subword-to-word label alignment
- Real-time named-entity extraction
- Confidence-based entity filtering
- Colored entity highlighting
- Entity confidence scores
- Entity start and end positions
- CSV export for detected entities
- Model-performance comparison dashboard
- Responsive and professional Streamlit interface
- Local Transformer loading without requiring internet access

---

## 🖼️ Application Preview

### Main Interface

The application provides a professional interface for entering English text, adjusting the confidence threshold, and viewing model information.

![NER Application Home](assets/screenshots/app_home.png)

### Entity Extraction Results

Detected entities are highlighted using different colors according to their entity type.

![NER Extraction Results](assets/screenshots/app_results.png)

---

## 🏆 Model Performance

All models were evaluated on the CoNLL-2003 test split using entity-level precision, recall, and F1 score.

| Model | Precision | Recall | Test F1 |
|---|---:|---:|---:|
| LSTM | 74.07% | 78.90% | 76.41% |
| BiLSTM | 82.84% | 84.19% | 83.51% |
| BiLSTM + CRF | 84.47% | 84.74% | 84.60% |
| **DistilBERT** | **88.20%** | **89.14%** | **88.67%** |

> **DistilBERT achieved the strongest overall performance with a Test F1 score of 88.67%.**

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

| Split | Number of Sentences |
|---|---:|
| Training | 14,041 |
| Validation | 3,250 |
| Test | 3,453 |

### Dataset Statistics

- Training tokens: **203,621**
- Validation tokens: **51,362**
- Test tokens: **46,435**
- Training vocabulary: **23,623 words**
- Maximum sequence length: **128 tokens**
- Number of NER labels: **9**
- Invalid IOB transitions: **0**

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

---

## 🏗️ Model Architectures

### 1. LSTM

The first model combines pre-trained word embeddings with character-level CNN features and a unidirectional LSTM.

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

The BiLSTM processes each sentence in both forward and backward directions, allowing every token to use contextual information from both sides.

```text
Word Features + Character Features
                ↓
        Bidirectional LSTM
                ↓
        Token Classification
```

### 3. BiLSTM + CRF

The CRF layer models dependencies between neighboring output labels and improves entity-boundary consistency.

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
- Highlight detected entities
- Adjust the minimum confidence threshold
- View confidence scores
- Inspect entity start and end positions
- Compare the performance of trained models
- Export detected entities as a CSV file
- Review dataset and architecture information
- Load ready-to-use example sentences

### Example Input

```text
Geoffrey Hinton worked at Google and received the Turing Award for his contributions to artificial intelligence.
```

### Example Output

| Entity | Type |
|---|---|
| Geoffrey Hinton | PER |
| Google | ORG |
| Turing Award | MISC |

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
├── assets/
│   └── screenshots/
│       ├── app_home.png
│       └── app_results.png
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

### 1. Clone the Repository

```bash
git clone https://github.com/abdelrahman-ahmed03/named-entity-recognition.git
cd named-entity-recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Linux or macOS

```bash
source .venv/bin/activate
```

### 4. Install the Required Packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Model Weights

Large model checkpoint files are excluded from Git using `.gitignore`.

The following files are not included in the repository:

```text
models/**/*.pt
models/**/*.pth
models/**/*.bin
models/**/*.safetensors
```

To run the Streamlit application, place the fine-tuned Transformer checkpoint inside:

```text
models/transformer/best_model/
```

The Transformer model directory should contain:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

Traditional model checkpoints should be placed inside:

```text
models/lstm/
models/bilstm/
models/bilstm_crf/
```

---

## ▶️ Run the Application

Run the following command from the project root:

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

Expected output:

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
| Programming Language | Python |
| Deep Learning | PyTorch |
| Transformer Models | Hugging Face Transformers, DistilBERT |
| Traditional NLP | GloVe, Character CNN, LSTM, BiLSTM, CRF |
| Evaluation | Seqeval, Scikit-learn |
| Data Processing | Pandas, NumPy, Hugging Face Datasets |
| User Interface | Streamlit |
| Visualization | Matplotlib, Streamlit Charts |
| Development | Jupyter Notebook, VS Code, Google Colab |

---

## 📈 Training Environment

The models were trained using:

- Google Colab
- NVIDIA Tesla T4 GPU
- PyTorch
- Hugging Face Transformers
- Maximum sequence length: 128
- Traditional model batch size: 32
- Transformer batch size: 16
- DistilBERT training epochs: 3

---

## 🔍 Evaluation Strategy

The project uses **entity-level evaluation** instead of relying only on token-level accuracy.

The main evaluation metrics include:

- Precision
- Recall
- Micro F1 score
- Macro F1 score
- Weighted F1 score
- Per-entity classification reports

Entity-level evaluation provides a more meaningful measurement of complete entity detection and boundary accuracy.

---

## 🚀 Future Improvements

- Add multilingual Named Entity Recognition
- Add batch file processing
- Support PDF and document uploads
- Build a REST API using FastAPI
- Deploy the Streamlit application publicly
- Add model quantization for faster CPU inference
- Train on additional domain-specific datasets
- Add experiment tracking
- Add Docker support
- Add automated tests and CI/CD

---

## 👨‍💻 Author

### Abdelrahman Ahmed

**AI Engineer & Data Scientist**

Machine Learning and Natural Language Processing enthusiast focused on building complete AI systems from data preparation and model training to evaluation and deployment.

[![GitHub](https://img.shields.io/badge/GitHub-abdelrahman--ahmed03-181717?style=for-the-badge&logo=github)](https://github.com/abdelrahman-ahmed03)

---

## ⭐ Support

If this project helped you, consider giving the repository a star.

<div align="center">

### Built with PyTorch, Hugging Face Transformers, and Streamlit

</div>
