# Deep Learning for Multi-Task Sentiment Analysis

This project implements a **Multi-Input / Multi-Output Deep Learning model** to analyze hotel reviews.
Unlike standard sentiment analysis, this model processes heterogeneous data sources (textual reviews + categorical metadata) to simultaneously predict two distinct targets: a continuous score (Regression) and a binary sentiment label (Classification).

## Model Architecture
The neural network is designed to handle **mixed data types**:
1.  **NLP Branch:** Processes the review text using **Word Embeddings** (and LSTM/GRU/Dense layers) to extract semantic features.
2.  **Metadata Branch:** Processes categorical features (e.g., Hotel ID, Date) using **Entity Embeddings** or normalization techniques.
3.  **Concatenation:** The features from both branches are merged into a dense feature vector.
4.  **Dual Output Heads:**
    * **Head A (Regression):** Predicts the numerical score (0-10) using a linear activation.
    * **Head B (Classification):** Predicts the sentiment polarity (Good/Bad) using a Sigmoid activation.

## Objectives
* Demonstrate proficiency in **Deep Learning** architectures beyond simple feed-forward networks.
* Implement **Multi-Task Learning** to optimize a single model for conflicting objectives (minimizing MSE for score and Binary Crossentropy for sentiment).
* Handle real-world noisy text data combined with structured features.

## Results Snapshot
The model outputs a dual prediction for each review.
*Example from Test Set:*
> **Review:** *"The location was excellent but the room was noisy."*
> * **Predicted Score:** 7.4/10 (Regression)
> * **Predicted Sentiment:** Positive (Probability: 0.85) (Classification)

## 🛠️ Technologies & Libraries
* **Python**
* **Deep Learning Framework:** TensorFlow / Keras (or PyTorch)
* **Data Processing:** Pandas, NumPy, Scikit-learn (for preprocessing/encoding)
* **NLP:** Tokenization, Padding, Embeddings
