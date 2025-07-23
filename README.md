# 🌸 Mood2Flowers

**Mood2Flowers** is a custom Transformer-based text classification model designed for the **Build Your Own Transformer from Scratch** Datathon. Given a user's emotional state, occasion, or personal description, the model predicts a suitable **flower bouquet composition**.

This project builds a Transformer **from scratch**, without relying on pre-built encoder/decoder classes from libraries like HuggingFace. It includes a custom tokenizer, positional encoding, multi-head attention, and training pipeline using PyTorch.

---

## 📝 Task Definition

> **Objective:** Predict the flower composition (label) based on user input describing a mood, an occasion, or a short text.

> **Input:** A sentence (e.g., "I feel lonely today", "Anniversary surprise", "Lost a loved one")  
> **Output:** A flower combination or bouquet category (e.g., "lavender and white lilies")

---

## 🏗️ Model Architecture

We implemented the core Transformer encoder modules manually:

- Custom **Tokenizer** & vocabulary builder
- **Embedding Layer**
- **Positional Encoding**
- **Multi-Head Attention**
- **Feedforward Layer**
- **Residual Connections** with Layer Normalization
- Final **classification head**

All components were implemented using **PyTorch** without any high-level shortcuts.

> 📂 Code structure is modularized inside the `src/` directory.

---

## 📚 Dataset

The dataset is a `.csv` file with the following columns:

| text                        | label                    |
|----------------------------|--------------------------|
| "I feel lost and hopeless" | "lavender, white roses"  |
| "Birthday of my mom"       | "pink tulips, daisies"   |

- We built a vocabulary from the training data
- Labels are encoded using `LabelEncoder`

---

## ⚙️ Preprocessing

- Lowercasing
- Tokenization using our custom tokenizer
- Padding/truncating sequences
- Splitting data into train/validation/test

---

## 🏋️ Training

Run the training with:

```bash
python src/train.py
