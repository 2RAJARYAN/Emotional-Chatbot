# Head: A Modular Emotion-Aware Middleware for LLM-Powered Chatbots

## 🌟 Overview

Head is a modular middleware designed to make any Large Language Model (LLM) emotionally aware without retraining the LLM itself.  
It acts as a plug-and-play emotion extraction component that processes user input, detects emotional states, and injects this metadata into the chatbot prompt pipeline.

This repository contains:

- 🧠 **Head Module** — A BERT/RoBERTa-based multi-label emotion classifier trained on GoEmotions (27 emotions).
- 💬 **Emotion-Aware Chat Application** — A Streamlit UI that integrates the Head module with an LLaMA-based chatbot using Ollama for real-time empathetic conversation.

## 📁 Model Checkpoints

- **sst2/** — Fine-tuned sentiment model + hyperparameter optimization (source task).  
- **goemotions/** — Final fine-tuned multi-label emotion model (target task).

---

## 🚀 Features

✔ Multi-label Emotion Detection (GoEmotions: 27 classes)  
✔ Transfer Learning (SST-2 → GoEmotions)  
✔ Hyperparameter tuning with Optuna (Bayesian search)  
✔ Cleaned Streamlit chatbot with memory  
✔ Emotion-conditioned prompting for LLaMA (Ollama)  
✔ Fully modular — LLM and emotion model are independent  
✔ Ready-to-deploy structure  

---

## 📂 Repository Structure


---

## 🏗️ System Architecture

### Pipeline

User Input
↓
Emotion Classifier (Head Module - RoBERTa)
↓
Detected Emotions + Confidence Scores
↓
Prompt Builder (adds emotional metadata)
↓
LLaMA (via Ollama)
↓
Empathetic Response
↓
Streamlit UI



---

## 🧠 Emotion Classification

### Datasets

| Dataset     | Purpose                        | Format            |
|-------------|--------------------------------|-------------------|
| SST-2       | Transfer learning source task  | Binary sentiment  |
| GoEmotions | Final fine-tuning              | 27 emotions       |

### Model Training

- Base architecture: **RoBERTa-base**  
- Loss: **BCEWithLogitsLoss** (multi-label)  
- Tokenization: `max_length=128`  
- Label encoding: **MultiLabelBinarizer**  
- Mixed precision training enabled  

---

## 🔧 Transfer Learning Procedure

1. Fine-tune RoBERTa on **SST-2**  
2. Hyperparameter optimization using **Optuna**  
3. Load best checkpoint  
4. Re-initialize final classifier for **27 emotion labels**  
5. Fine-tune on **GoEmotions**  
6. Tune threshold (0.1–0.9 grid) for best macro F1  

---

## 📊 Benchmarks

### SST-2 Results

| Metric     | Value     |
|------------|-----------|
| Accuracy   | ~94%      |
| Best LR    | 2.28e-5   |
| Batch Size | 16        |
| Epochs     | 4         |

### GoEmotions Results

| Model                       | Macro F1 | Micro F1 |
|-----------------------------|----------|-----------|
| Baseline (no transfer)      | ~0.42    | ~0.60     |
| Transfer (SST-2 → GoEmotions) | ~0.45    | ~0.62     |

✨ Transfer learning gave a measurable improvement.

---

## 💬 Chatbot (Streamlit + Ollama)

The chatbot uses detected emotions to modulate:
- Tone (warm, supportive, calm)
- Word choice
- Length
- Follow-up questions

Example emotion metadata:

Detected emotions: sadness (0.82), confusion (0.41)


Prompt sent to LLaMA:



You are an empathetic assistant. Use the user's detected emotions to adjust tone.
Conversation history:
...
Respond to the latest user message.


---

## 🛠️ Installation

### 1. Create Conda environment

conda create -n env_name
conda activate env_name


### 2. Install dependencies



pip install -r requirements.txt



Minimal requirements:

- transformers  
- torch  
- scikit-learn  
- streamlit  
- datasets  
- optuna  
- langchain-ollama  
- requests  


*(Update paths inside scripts as needed.)*

---

## 🧱 Future Improvements

- Add conversation-level emotion tracking  
- Use cross-encoders for context-aware emotion detection  
- Add speech (audio emotion,video emotion) → multimodal Head  
- Compression/distillation for mobile deployment  

---

## 🤝 Contributing

Pull requests are welcome!  
Open an issue if you want to discuss new features.

---

## 📜 License

MIT License (modify as needed)



