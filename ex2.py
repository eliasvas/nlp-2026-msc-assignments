import numpy as np
from collections import Counter
import re

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

set_splits = [0.5, 0.25, 0.25]
N_Doc = 2000
y = np.asarray(y)

n_train = int(N_Doc * set_splits[0])
n_dev = int(N_Doc * set_splits[1])
x_train, y_train = x[:n_train], y[:n_train]
x_dev, y_dev = x[n_train:n_train+n_dev], y[n_train:n_train+n_dev]
x_test, y_test   = x[n_train+n_dev:N_Doc], y[n_train+n_dev:N_Doc]

def tokenize(doc):
    return re.findall(r"[a-z']+", doc.lower())

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_tf(vocab, vword_to_idx, doc):
  new_tf = np.zeros(len(vocab))
  words = tokenize(doc)
  for word in words:
    j = vword_to_idx.get(word)
    if j is not None:
      new_tf[j] += 1

  return new_tf

def calc_tf_all(docs, vocab, vword_to_idx, k=7):
  tf = np.array([compute_tf(vocab, vword_to_idx, d) for d in docs])
  sum_tfs = tf.sum(axis=0)
  sel = np.argsort(sum_tfs)[-k:]
  sel_vocab = [vocab[i] for i in sel]
  sel_idx = {w_: i for i, w_ in enumerate(sel_vocab)}
  return tf[:, sel], sel_vocab, sel_idx


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


vocab = sorted(set(
  w.lower()
  for doc in x[0:int(N_Doc*set_splits[0])]
  for w in tokenize(doc)
  if w.lower() not in STOPWORDS
))
vword_to_idx = {w:idx for idx, w in enumerate(vocab)}

best_k, best_lambda_reg, best_acc = 0, 0, 0
for k in [10, 50, 200, 500]:
  X_train, vocab_k, idx_k = calc_tf_all(x_train, vocab, vword_to_idx, k)
  X_dev  = np.array([compute_tf(vocab_k, idx_k, d) for d in x_dev])

  for reg in [0.001, 0.01, 0.1]:
    w, b = train(X_train, y_train, epochs=3000, lr=1, lambda_reg=reg)
    acc = evaluate(X_dev, y_dev, w, b)
    if acc > best_acc:
      best_acc = acc
      best_k = k
      best_lambda_reg = reg

#retrain with best k, reg, etc
X_train, best_vocab, best_idx = calc_tf_all(x_train, vocab, vword_to_idx, best_k)
w, b = train(X_train, y_train, epochs=100, lr=0.1, lambda_reg=best_lambda_reg)



print("Best k:", best_k)
print("Best lambda_reg:", best_lambda_reg)
print("Accuracy:", acc)

# 3. Predict sentiment for new sentence
new_sentence = "This film is like very bad"
new_tf = compute_tf(best_vocab, best_idx, new_sentence)
new_class = classify(new_tf, w, b)
print(new_sentence, "->", new_tf, "->", best_vocab)
print("Test sentence:", new_sentence)
print("Predicted sentiment:", "Positive" if (new_class == 1) else "Negative")

new_class_majority = majority_classify(new_tf, w, b)
print("Predicted sentiment (majority):", "Positive" if (new_class_majority == 1) else "Negative")
