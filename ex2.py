### TEST
import numpy as np
from collections import Counter

STOPWORDS = set([
    "the", "is", "a", "an", "and", "or", "but",
    "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "our", "their",
    "to", "of", "in", "on", "at", "for", "with",
    "as", "by", "from", "about",
    "was", "were", "am", "are", "be", "been",
    "have", "has", "had", "do", "does", "did",
    "very", "just", "so", "too"
])


def split_dataset(x,y, set_splits, N_Doc, k):
  train_start, train_end = 0, int(N_Doc*set_splits[0])
  vocab, vword_to_idx, tf = calc_vocab_and_tf(x[train_start:train_end], k)
  X_train = tf
  y_train = y[0:int(N_Doc*set_splits[0])]


  dev_start, dev_end = train_end, train_end + int(N_Doc*set_splits[1])
  X_dev = [compute_tf(vocab, vword_to_idx, x[doc_idx]) for doc_idx in range(dev_start, dev_end)]
  y_dev = y[dev_start:dev_end]


  test_start, test_end = dev_end, dev_end + int(N_Doc*set_splits[2])
  X_test = [compute_tf(vocab, vword_to_idx, x[doc_idx]) for doc_idx in range(test_start, test_end)]
  y_test = y[test_start:test_end]
  return X_train, y_train, X_dev, y_dev, X_test, y_test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_tf(vocab, vword_to_idx, doc):
  new_tf = np.zeros(len(vocab))
  words = doc.split(' ')
  for word in words:
    word = word.lower()
    if word in vocab:
      new_tf[vword_to_idx[word]] += 1
  new_tf /= len(words)
  return new_tf

def calc_vocab_and_tf(docs, k = 7):
  # 0. Produce the vocabulary and auxillary data
  vocab = sorted(set(
    w.lower()
    for doc in docs
    for w in doc.split()
    if w.lower() not in STOPWORDS
  ))
  vword_to_idx = {w:idx for idx, w in enumerate(vocab)}

  # 1. Produce the TF vectors for each word
  tf = np.zeros((len(docs), len(vocab)))
  words_per_doc = np.zeros(len(docs))
  for doc_idx, doc in enumerate(docs):
    tf[doc_idx] = compute_tf(vocab, vword_to_idx, doc)

  # 2. Do feature selection (+ recalc tf/vocab/vword_to_idx)
  #sum_tfs = tf.sum(axis=0)
  #features_selected = np.argsort(sum_tfs)[-k:]


  positive_tfs = tf[y[:len(tf)] == 1].mean(axis=0)
  negative_tfs = tf[y[:len(tf)] == 0].mean(axis=0)
  sentiment_scores = np.abs(positive_tfs - negative_tfs)
  features_selected = np.argsort(sentiment_scores)[-k:]

  tf = tf[:, features_selected]
  vocab = np.array(vocab)[features_selected]
  vword_to_idx = {w:idx for idx, w in enumerate(vocab)}

  return vocab, vword_to_idx, tf


def train(X, y, epochs=1000, lr=0.1, lambda_reg = 0.01):
  w = np.zeros(X.shape[1])
  b = 0.0
  for epoch in range(epochs):
    z = np.dot(X, w) + b
    predictions = sigmoid(z)

    dw = np.dot(X.T, (predictions - y)) / X.shape[0]
    # regualrize
    dw += (lambda_reg / X.shape[0]) * w

    db = np.mean(predictions - y)

    w -= lr * dw
    b -= lr * db
  return w, b

def classify(X, w, b):
  pred = sigmoid(np.dot(X, w) + b)
  pred_labels = (pred >= 0.5).astype(int)
  return pred_labels
def evaluate(X, y, w, b):
  pred_labels = classify(X,w,b)
  return np.mean(pred_labels == y)

def majority_classify(X, w, b):
  class_counts = np.bincount(y)
  positive_pct = class_counts[1]/len(x)
  return (positive_pct >= 0.5).astype(int)


set_splits = [0.5, 0.25, 0.25]
N_Doc = 1000

best_k, best_lambda_reg, best_acc = 0, 0, 0
for k in [1, 20, 100]:
  X_train, y_train, X_dev, y_dev, X_test, y_test = split_dataset(x, y, set_splits, N_Doc, k)

  train_start, train_end = 0, int(N_Doc*set_splits[0])
  vocab, vword_to_idx, tf = calc_vocab_and_tf(x[train_start:train_end], k)

  for reg in [0.001, 0.01, 0.1]:
    w, b = train(X_train, y_train, epochs=100, lr=0.1, lambda_reg=reg)

    acc = evaluate(X_dev, y_dev, w, b)
    if acc > best_acc:
      best_acc = acc
      best_k = k
      best_lambda_reg = reg

print("Best k:", best_k)
print("Best lambda_reg:", best_lambda_reg)
print("Accuracy:", acc)

# 3. Predict sentiment for new sentence
new_sentence = "This film is like very good"
new_tf = compute_tf(vocab, vword_to_idx, new_sentence)
new_class = classify(new_tf, w, b)
print(new_sentence, "->", new_tf, "->", vocab)
print("Test sentence:", new_sentence)
print("Predicted sentiment:", "Positive" if (new_class == 1) else "Negative")

new_class_majority = majority_classify(new_tf, w, b)
print("Predicted sentiment (majority):", "Positive" if (new_class == 1) else "Negative")


