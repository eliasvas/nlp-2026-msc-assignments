#############################
## Sample N-Gram model trained on genesis corpus

nltk.download('genesis')
nltk.download('punkt')
nltk.download('punkt_tab')

import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import genesis
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = {
    "additional_special_tokens": ["START", "END"]
}
tokenizer.add_special_tokens(special_tokens)
start_id = tokenizer.convert_tokens_to_ids("START")
end_id = tokenizer.convert_tokens_to_ids("END")



def corpus_from_sentence_slice(sentences, start, end):
  sents = sentences[start:end]
  corpus = [] # List of token ids for each sentence augmented with <s> </s>
  for sent in sents:
    text = " ".join(sent)
    ids = [tokenizer.convert_tokens_to_ids("START")] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("END")]
    corpus.append(ids)
  return corpus

def bigram_make(corpus):
  vocab = np.unique([id for c in corpus for id in c])
  bigram = {
    "vocab" : vocab,
    "freqs" : np.zeros((vocab.size+1, vocab.size+1), dtype=int),
    "probs" : np.zeros((vocab.size+1, vocab.size+1), dtype=float)
  }
  vocab_index = {token_id: i for i, token_id in enumerate(vocab)}
  for sent in corpus:
    for i in range(len(sent)-1):
      tok0_vid = vocab_index.get(sent[i], vocab.size)
      tok1_vid = vocab_index.get(sent[i+1], vocab.size)
      bigram["freqs"][tok0_vid][tok1_vid]+=1
  return bigram

def eval_bigram(bigram_dict, corpus, alpha):
  freqs = bigram_dict["freqs"]
  vocab = bigram_dict["vocab"]
  V = bigram_dict["vocab"].size

  N = 0
  log_prob_sum = 0

  vocab_index = {token_id: i for i, token_id in enumerate(vocab)}
  for sentence in corpus:
    for i in range(len(sentence)-1):
      if sentence[i] == start_id:
        continue
      tok0_vid = vocab_index.get(sentence[i], vocab.size)
      tok1_vid = vocab_index.get(sentence[i+1], vocab.size)

      bi_count = freqs[tok0_vid][tok1_vid]
      uni_count = np.sum(freqs[tok0_vid])
      P = (bi_count + alpha) / (uni_count + alpha * len(tokenizer))
      bigram_dict["probs"][tok0_vid][tok1_vid] = P

      log_prob_sum += np.log2(P)
      N += 1
  cross_entropy = -log_prob_sum / N if N > 0 else float("inf")
  perplexity = 2 ** cross_entropy
  return cross_entropy, perplexity, N

def autocomplete_bigram(bigram_dict, text):
  freqs = bigram_dict["freqs"]
  probs = bigram_dict["probs"]
  vocab = bigram_dict["vocab"]
  ids = [tokenizer.convert_tokens_to_ids("START")] + tokenizer.encode(text, add_special_tokens=False)
  vocab_index = {token_id: i for i, token_id in enumerate(vocab)}
  while ids[-1] != end_id:
    tok0_vid = vocab_index.get(ids[-1], vocab.size)
    f = freqs[tok0_vid]
    p = probs[tok0_vid]
    if f.sum() == 0:
      break

    temperature = 0.6
    # this is to skip the invalid index..
    p = p[:-1]
    sampling_probs = np.exp(np.log(p + 1e-10) / temperature)
    sampling_probs /= sampling_probs.sum()
    next_tok_idx = np.random.choice(len(vocab), p=sampling_probs)

    ids.append(int(vocab[next_tok_idx]))
  return ids


def test_bigram():
  ####################################
  ## Produce training sentences
  all_sentences = genesis.sents()[:1000]

  train_pcts = [0.2, 0.5, 0.8, 0.9, 1.0]
  entropies_per_train_pct = []
  perplexities_per_train_pct = []  

  for train_pct in train_pcts:
    ####################################
    ## Assemble the corpus
    sent_count = len(all_sentences)
    print("Training with ", int((sent_count/2)*train_pct), "sentences")
    test_corpus = corpus_from_sentence_slice(all_sentences, 0, int(sent_count/4))
    dev_corpus   = corpus_from_sentence_slice(all_sentences, int(sent_count/4), int(sent_count/2))
    train_corpus = corpus_from_sentence_slice(all_sentences, int(sent_count/2), int(sent_count/2 + (sent_count/2)*train_pct))

    ####################################
    ## Make the bigram model
    bigram = bigram_make(train_corpus)
    vocab = bigram["vocab"]

    #################################
    ## Use the dev set to find the best alpha
    possible_alphas = [0.0001, 0.0003, 0.001, 0.009, 0.1, 0.5, 0.9]
    best_alpha = possible_alphas[-1]
    best_ce = float("inf")
    for alpha in possible_alphas:
      CE,P,N = eval_bigram(bigram, dev_corpus, alpha)
      print("With alpha: ", alpha, ", CE=", CE)
      if CE < best_ce:
        best_ce = CE
        best_alpha = alpha
    print("Selected alpha: ", best_alpha)

    #################################
    ## Evaluate every sentence in the test corpus
    CE,P,N = eval_bigram(bigram, test_corpus, best_alpha)
    entropies_per_train_pct.append(CE)
    perplexities_per_train_pct.append(P)

  #################################
  ## Show the training-size -> entropy graph
  plt.figure(figsize=(5, 2.7), layout='constrained')
  plt.xlabel("training set slice (%)")
  plt.ylabel("cross-entropy")
  plt.title("Training w/ different sizes")
  plt.plot(train_pcts, entropies_per_train_pct)


  #################################
  ## Show the training-size -> perplexity graph
  plt.figure(figsize=(5, 2.7), layout='constrained')
  plt.xlabel("training set slice (%)")
  plt.ylabel("perplexity")
  plt.title("Training w/ different sizes")
  plt.plot(train_pcts, perplexities_per_train_pct)

  #################################
  ## Trying autocomplete
  texts = [
      "I would like to",
      "Hello there",
      "I can't find my"
  ]
  for text in texts:
    ids = autocomplete_bigram(bigram, text)
    print(text, " -> ", tokenizer.decode(ids, skip_special_tokens=True))


test_bigram()

