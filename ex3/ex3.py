%pip install --upgrade pip --quiet
%pip install notebook jupyter --quiet
%pip install nltk --quiet
%pip install spacy --quiet
%pip install numpy --quiet
%pip install scikit-learn --quiet
%pip install pandas --quiet
%pip install matplotlib seaborn --quiet
%pip install umap-learn --quiet
%pip install wget --quiet


import wget
url = "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
filename = wget.download(url)
#print(filename)
!tar xzf review_polarity.tar.gz
from sklearn.datasets import load_files
movies = load_files(container_path="txt_sentoken", encoding="utf-8")
x, y = movies.data, movies.target

# The above part is taken straight from the lab..
import numpy as np
from collections import Counter
import re

import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score

# yes this isnt the best solution..
STOPWORDS = set("""
the is a an and or but
this that these those
i you he she it we they
my your his her our their
to of in on at for with
as by from about
was were am are be been
have has had do does did
very just so too
""".split())

set_splits = [0.5, 0.25, 0.25]
N_Doc = 1000
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


def train_logistic(X, y, epochs=1000, lr=0.1, lambda_reg = 0.01):
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

def classify_mlp(X, w, b):
  pred = sigmoid(np.dot(X, w) + b)
  pred_labels = (pred >= 0.5).astype(int)
  return pred_labels

def evaluate_mlp(X, y, w, b):
  pred_labels = classify_logistic(X,w,b)
  return np.mean(pred_labels == y)

def classify_logistic(X, w, b):
  pred = sigmoid(np.dot(X, w) + b)
  pred_labels = (pred >= 0.5).astype(int)
  return pred_labels

def evaluate_logistic(X, y, w, b):
  pred_labels = classify_logistic(X,w,b)
  return np.mean(pred_labels == y)

def classify_majority(X, w, b):
    majority = np.bincount(y_train).argmax()
    return np.full(X.shape[0], majority)

def calc_macro_f1(y_true, y_pred):
  f1_scores = []
  for cls in [0,1]:
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    acc = np.mean(y_pred == y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    acc = np.mean(y_pred == y_true)
    f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)
    f1_scores.append(f1)
  return np.mean(f1_scores)

class Quality_Metrics:
  precision : float
  recall : float
  accuracy : float
  f1 : float
  pr_auc : float
  def __init__(self, precision, recall, accuracy, f1, pr_auc):
    self.precision = precision
    self.recall = recall
    self.accuracy = accuracy
    self.f1 = f1
    self.pr_auc = pr_auc
  def __add__(self, other):
    return Quality_Metrics(
        self.precision + other.precision,
        self.recall + other.recall,
        self.accuracy + other.accuracy,
        self.f1 + other.f1,
        self.pr_auc + other.pr_auc
    )
  def __truediv__(self, other):
    return Quality_Metrics(
        self.precision / other,
        self.recall / other,
        self.accuracy / other,
        self.f1 / other,
        self.pr_auc / other
    )

vocab = sorted(set(
  w.lower()
  for doc in x[0:int(N_Doc*set_splits[0])]
  for w in tokenize(doc)
  if w.lower() not in STOPWORDS
))
vword_to_idx = {w:idx for idx, w in enumerate(vocab)}

# train logistic
best_k, best_lambda_reg, best_acc = 0, 0, 0
for k in [10, 50, 200, 500]:
  X_train, vocab_k, idx_k = calc_tf_all(x_train, vocab, vword_to_idx, k)
  X_dev  = np.array([compute_tf(vocab_k, idx_k, d) for d in x_dev])

  for reg in [0.001, 0.01, 0.1]:
    w, b = train_logistic(X_train, y_train, epochs=100, lr=1, lambda_reg=reg)
    acc = evaluate_logistic(X_dev, y_dev, w, b)
    if acc > best_acc:
      best_acc = acc
      best_k = k
      best_lambda_reg = reg

print("Logistic best k:", best_k)
print("Logistic best lambda_reg:", best_lambda_reg)

# Calculate the final train/test/dev sets
X_train, best_vocab, best_idx = calc_tf_all(x_train, vocab, vword_to_idx, best_k)
X_dev  = np.array([compute_tf(best_vocab, best_idx, d) for d in x_dev])
X_test = np.array([compute_tf(best_vocab, best_idx, d) for d in x_test])
# Find the weight for the best lambda and k
w, b = train_logistic(X_train, y_train, epochs=100, lr=1, lambda_reg=best_lambda_reg)


for set_name, my_set, my_y in [
    ("train", X_train, y_train),
    ("dev", X_dev, y_dev),
    ("test", X_test, y_test),
]:
  metrics = []
  for cls in [0,1]:
    for cidx, my_classify in enumerate([classify_logistic, classify_majority, classify_mlp]):
      predictions = my_classify(my_set, w, b)
      tp = np.sum((predictions == cls) & (my_y == cls))
      fp = np.sum((predictions == cls) & (my_y != cls))
      fn = np.sum((predictions != cls) & (my_y == cls))
      acc = np.mean(predictions == my_y)
      precision = tp / (tp + fp) if tp + fp > 0 else 0
      recall = tp / (tp + fn) if tp + fn > 0 else 0
      acc = np.mean(predictions == my_y)
      f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

      #This ok?
      scores = (predictions == cls).astype(int)
      pr_auc = average_precision_score(
          (my_y == cls).astype(int), scores
      )

      m = Quality_Metrics(precision, recall, acc, f1, pr_auc)
      metrics.append(
          ['+' if cls == 1 else '-',
          'majority' if 'majority' in my_classify.__name__ else 'logistic' if 'logistic' in my_classify.__name__ else 'mlp', m]
      )

  total = 7*12
  pad = (total - len(set_name)) // 2
  print('-' * pad, end="");print(set_name, end="");print('-' * pad)
  print(f"{'class':>12}{'classifier':>12}{'acc':>12}{'prec':>12}{'recall':>12}{'f1':>12}{'auc':>12}")

  avgs = [Quality_Metrics(0,0,0,0,0), Quality_Metrics(0,0,0,0,0), Quality_Metrics(0,0,0,0,0)]
  for metric in metrics:
    m = metric[2]
    avgs[0 if metric[1] == 'majority' else 1] += m
    print(f"{metric[0]:>12}{metric[1]:>12}{m.accuracy:>12.2f}{m.precision:>12.2f}{m.recall:>12.2f}{m.f1:>12.2f}{m.pr_auc:>12.2f}")
  for idx, avg in enumerate(avgs):
    avg /= 2 # 1 for (+) 1 for (-) for each classifier
    print(f"{'macro':>12}{'majority' if idx == 0 else 'logistic' if idx == 1 else 'mlp':>12}{avg.accuracy:>12.2f}{avg.precision:>12.2f}{avg.recall:>12.2f}{avg.f1:>12.2f}{avg.pr_auc:>12.2f}")
  print('-' * total)



mlp_train = []
mlp_dev = []

logistic_train = []
logistic_dev = []

majority_train = []
majority_dev = []

percentages = [0.1, 0.3, 0.7, 1]
for percentage in percentages:
  X_subset = X_train[:int(percentage * len(X_train))]
  y_subset = y_train[:int(percentage * len(y_train))]
  w, b = train_logistic(X_subset, y_subset, epochs=100, lr=1, lambda_reg=best_lambda_reg)

  # mlp
  train_pred = classify_mlp(X_subset, w, b)
  dev_pred = classify_mlp(X_dev, w, b)
  mlp_train.append(calc_macro_f1(y_subset, train_pred))
  mlp_dev.append(calc_macro_f1(y_dev, dev_pred))

  # logistic
  train_pred = classify_logistic(X_subset, w, b)
  dev_pred = classify_logistic(X_dev, w, b)
  logistic_train.append(calc_macro_f1(y_subset, train_pred))
  logistic_dev.append(calc_macro_f1(y_dev, dev_pred))

  # majority
  majority = np.bincount(y_subset).argmax()
  train_pred = np.full(len(y_subset), majority)
  dev_pred = np.full(len(y_dev), majority)
  majority_train.append(calc_macro_f1(y_subset, train_pred))
  majority_dev.append(calc_macro_f1(y_dev, dev_pred))


# mlp plot
plt.figure(figsize=(7, 5))
plt.plot(percentages, mlp_train, marker='o', label='Training')
plt.plot(percentages, mlp_dev, marker='o', label='Development')
plt.title("MLP learning curve")
plt.xlabel("train split")
plt.ylabel("macro_f1")
plt.xticks(percentages, ['10%', '30%', '70%', '100%'])
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

# logistic plot
plt.figure(figsize=(7, 5))
plt.plot(percentages, logistic_train, marker='o', label='Training')
plt.plot(percentages, logistic_dev, marker='o', label='Development')
plt.title("Logistic learning curve")
plt.xlabel("train split")
plt.ylabel("macro_f1")
plt.xticks(percentages, ['10%', '30%', '70%', '100%'])
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

# majority plot
plt.figure(figsize=(7, 5))
plt.plot(percentages, majority_train, marker='o', label='Training')
plt.plot(percentages, majority_dev, marker='o', label='Development')
plt.title("Majority learning curve")
plt.xlabel("train split")
plt.ylabel("macro_f1")
plt.xticks(percentages, ['10%', '30%', '70%', '100%'])
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()


"""
> Dataset: I used the cornell review polarity dataset that has binary classifiaction of film reviews.
  As can be seen the graphs are the same as for the accompanying lab's. We have two classes (+) and (-)
  with 1000 documents for each and similar word distributions for each class.
> On how this whole thing works:
  1.) We preprocess the reviews to remove common stopwords - for speed and so that we don't choose them for the vocab.
  2.) For vocabulary calculation initially we just get every word in the documents (we will chop this up with k)
  3.) Now we have to train with X_train and evaluate on X_dev so that we can tune the hyperparameters (lamda_reg and k).
  4.) For every hyperparameter combination we train using logistic regression (sigmoid) with regularization (helped a lot).
  5.) After training we evaluate on the development set and compute/store the combination k+lambda_reg that gives the best accuracy.
  6.) Now we have the best possible combination so we train once more with the optimal hyperparameters to learn the final weights / bias.
  7.) With this classifier and using a dumb majority classifier for comparison, we compute the quality metrics (prec,recall,f1,pr_auc) for all the three sets.
  8.) We do a textual representation for each classifier/class in each training set and then compute also the macro (averaged) results per classifier/class.
  9.) Finally we calculate the learning curves by using every increasing slices of the training set, and we plot the dev and train sets for both classifiers.
"""
