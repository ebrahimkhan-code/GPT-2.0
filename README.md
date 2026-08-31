# GPT-2.0 — From-Scratch Transformer Language Model

A from-scratch implementation of a **GPT-style language model using PyTorch**.

This project is designed to understand how modern Generative Pre-trained Transformer (GPT) models work internally by implementing the major components step by step instead of relying on a high-level pretrained GPT implementation.

## 🚀 Project Overview

The project follows the core ideas behind the original GPT architecture:

- Character-level tokenization
- Vocabulary creation
- Training/validation data split
- Bigram language model
- Token and positional embeddings
- Self-attention
- Scaled dot-product attention
- Multi-head attention
- Feed-forward neural networks
- Residual connections
- Layer normalization
- Transformer blocks
- GPT-style language model
- Autoregressive text generation
- Model training and evaluation

The implementation is intended as an educational project for understanding the architecture behind GPT and Transformer-based language models.

## 🧠 Architecture

The model follows this general pipeline:

```text
Input Text
    │
    ▼
Character Tokenization
    │
    ▼
Token IDs
    │
    ▼
Token Embeddings + Positional Embeddings
    │
    ▼
Transformer Blocks
    │
    ├── Multi-Head Self-Attention
    │
    ├── Residual Connection
    │
    ├── Layer Normalization
    │
    ├── Feed-Forward Network
    │
    └── Residual Connection
    │
    ▼
Language Modeling Head
    │
    ▼
Logits
    │
    ▼
Next-Token Prediction
    │
    ▼
Generated Text
```

## 📚 Learning Progression

The project builds the language model incrementally:

### 1. Bigram Language Model

A simple baseline that predicts the next token using only the current token.

### 2. Self-Attention

Introduces token-to-token communication so the model can use contextual information from previous tokens.

### 3. Multi-Head Attention

Runs multiple attention mechanisms in parallel, allowing the model to learn different relationships between tokens.

### 4. Feed-Forward Network

Adds a nonlinear transformation after attention.

### 5. Transformer Block

Combines:

- Multi-head self-attention
- Feed-forward network
- Residual connections
- Layer normalization

### 6. GPT Model

Combines multiple Transformer blocks into an autoregressive language model capable of generating text one token at a time.

## 🛠️ Technologies

- **Python**
- **PyTorch**
- **NumPy**
- **Jupyter Notebook / Python scripts**
- Transformer architecture
- Neural language modeling

## 📦 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd GPT-2.0
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the project does not contain a `requirements.txt`, install PyTorch manually:

```bash
pip install torch numpy
```

## ▶️ Running the Project

Run the main training script provided by the project:

```bash
python <main_script>.py
```

If the implementation is organized into notebooks, open the notebook and run the cells sequentially.

The training process learns to predict the next character/token from the provided training corpus. After training, the model can generate new text autoregressively.

## ⚙️ Important Hyperparameters

The implementation can be experimented with using parameters such as:

| Parameter | Purpose |
|---|---|
| `batch_size` | Number of sequences processed per training step |
| `block_size` | Maximum context length |
| `max_iters` | Number of training iterations |
| `learning_rate` | Optimizer learning rate |
| `eval_interval` | Frequency of evaluation |
| `eval_iters` | Number of batches used for evaluation |
| `n_embd` | Embedding dimension |
| `n_head` | Number of attention heads |
| `n_layer` | Number of Transformer blocks |
| `dropout` | Dropout probability |

Example configuration:

```python
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4

n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
```

These values can be adjusted depending on available CPU/GPU memory and the size of the training dataset.

## 🖥️ CPU / GPU

The project can automatically use CUDA when available:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

For larger models and faster training, an NVIDIA GPU with CUDA support is recommended.

The model can still be trained on a CPU with smaller hyperparameters.

## 📖 Core Concepts Demonstrated

### Token Embeddings

Each input token is converted into a learnable vector representation.

### Positional Embeddings

Since Transformers do not inherently understand token order, positional information is added to the token embeddings.

### Self-Attention

Attention allows each token to determine which previous tokens are important for predicting the next token.

Conceptually:

```text
Query × Key
     │
     ▼
Attention Scores
     │
     ▼
Softmax
     │
     ▼
Weighted Values
```

### Causal Masking

The model is autoregressive, so a token cannot look at future tokens during training.

```text
Token 1 → can see Token 1
Token 2 → can see Token 1, 2
Token 3 → can see Token 1, 2, 3
...
```

### Multi-Head Attention

Multiple attention heads allow the model to learn different contextual relationships simultaneously.

### Feed-Forward Network

The feed-forward layer provides additional nonlinear representation learning after attention.

### Residual Connections

Residual connections help information and gradients flow through deep Transformer networks.

### Layer Normalization

Layer normalization stabilizes the training process.

## 🧪 Experiments

This project is suitable for experimenting with:

- Different context lengths
- Different embedding dimensions
- Number of attention heads
- Number of Transformer layers
- Learning rates
- Batch sizes
- Dropout values
- Training iterations
- Different text datasets

You can compare how each change affects:

- Training loss
- Validation loss
- Generated text quality
- Training speed
- Memory usage

## 📝 Dataset

The model learns from a plain-text corpus.

The training data should be placed in the location expected by the training script, for example:

```text
input.txt
```

The implementation can construct a character vocabulary from the dataset and encode characters into integer token IDs.

## 📂 Typical Project Structure

```text
GPT-2.0/
│
├── input.txt
├── requirements.txt
├── *.py
├── *.ipynb
└── README.md
```

The exact structure may vary depending on the training stage and experiments included in the repository.

## 🎯 Project Goals

The main goals of this project are to:

1. Understand how GPT models work internally.
2. Learn the mathematics and implementation of self-attention.
3. Implement Transformer blocks using PyTorch.
4. Train an autoregressive language model from scratch.
5. Understand the relationship between tokens, embeddings, attention, and text generation.
6. Experiment with model architecture and training hyperparameters.

## 🔮 Future Improvements

Possible improvements include:

- Subword tokenization such as BPE
- Larger and cleaner training datasets
- Better text preprocessing
- Learning-rate scheduling
- Weight tying
- Mixed-precision training
- GPU optimization
- Checkpoint saving/loading
- Experiment tracking
- More Transformer layers
- Larger embedding dimensions
- Fine-tuning on domain-specific datasets
- Evaluation using standard language-model metrics

## 👨‍💻 Author

**Ebrahim Khan**

Computer Science student and AI/ML enthusiast.

## ⭐ Acknowledgements

This project is inspired by the educational work of **Andrej Karpathy** and the original Transformer/GPT research.

Important references:

- *Attention Is All You Need* — Vaswani et al.
- *Improving Language Understanding by Generative Pre-Training* — Radford et al.
- *Language Models are Unsupervised Multitask Learners* — Radford et al.

## 📄 License

This project is intended primarily for educational and research purposes. Add a specific open-source license such as MIT if you want to explicitly permit reuse and modification.
