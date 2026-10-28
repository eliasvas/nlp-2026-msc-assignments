#############################
## AutoTokenizer example
## https://huggingface.co/docs/transformers/main_classes/tokenizer

import nltk
import numpy as np
from collections import Counter, defaultdict
from nltk.corpus import brown
from transformers import AutoTokenizer

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

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
    "freqs" : np.zeros((vocab.size, vocab.size), dtype=int)
  }
  vocab_index = {token_id: i for i, token_id in enumerate(vocab)}
  for sent in corpus:
    for i in range(len(sent)-1):
      tok0_vid = vocab_index.get(sent[i], None)
      tok1_vid = vocab_index.get(sent[i+1], None)
      if tok0_vid is None or tok1_vid is None:
          continue
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
      tok0_vid = vocab_index.get(sentence[i], None)
      tok1_vid = vocab_index.get(sentence[i+1], None)
      if tok0_vid is None or tok1_vid is None:
          continue

      bi_count = freqs[tok0_vid][tok1_vid]
      uni_count = np.sum(freqs[tok0_vid])
      P = (bi_count + alpha) / (uni_count + alpha * vocab.size)

      log_prob_sum += np.log2(P)
      N += 1

      #print(tokenizer.convert_ids_to_tokens(ids[i]),"+", tokenizer.convert_ids_to_tokens(ids[i+1]), bi_count)
      #print(tokenizer.convert_ids_to_tokens(ids[i]), "has overall count:", uni_count)
      #print("P(", tokenizer.convert_ids_to_tokens(sentence[i]),"+", tokenizer.convert_ids_to_tokens(sentence[i+1]), ") = ", P)
  
  cross_entropy = -log_prob_sum / N if N > 0 else float("inf")
  perplexity = 2 ** cross_entropy
  return cross_entropy, perplexity, N


####################################
## Produce training sentences
nltk.download("brown")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = {
    "additional_special_tokens": ["START", "END"]
}
tokenizer.add_special_tokens(special_tokens)
all_sentences = brown.sents()[:200]

####################################
## Assemble the corpus
train_corpus = corpus_from_sentence_slice(all_sentences, 0, 100)
dev_corpus   = corpus_from_sentence_slice(all_sentences, 101, 150)
test_corpus  = corpus_from_sentence_slice(all_sentences, 151, 200)

####################################
## Make the bigram model
bigram = bigram_make(train_corpus)
vocab = bigram["vocab"]

#################################
## Use the dev set to find the best alpha
possible_alphas = [0.001, 0.009, 0.03, 0.1, 0.5, 0.8, 0.9]
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


