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
  for sent in corpus:
    for i in range(len(sent)-1):
      tok0_vid = np.where(vocab == sent[i])[0][0]
      tok1_vid = np.where(vocab == sent[i+1])[0][0]
      bigram["freqs"][tok0_vid][tok1_vid]+=1
  return bigram

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
test_corpus  = corpus_from_sentence_slice(all_sentences, 0, 100)

####################################
## Make the bigram model
bigram = bigram_make(train_corpus)
vocab = bigram["vocab"]

"""
####################################
## Find the propability for a test text (use other slice for testing!)
text = "The Fulton County Grand Jury said"
ids = [tokenizer.convert_tokens_to_ids("START")] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("END")]
print(ids)
for i in range(len(ids)-1):
    tok0_vid = np.where(vocab == ids[i])[0][0]
    tok1_vid = np.where(vocab == ids[i+1])[0][0]

    bi_count = bigram["freqs"][tok0_vid][tok1_vid]
    uni_count = np.sum(bigram["freqs"][tok0_vid])

    alpha = 0.009
    P = (bi_count + alpha) / (uni_count + alpha * vocab.size)
    #P = (bi_count) / (uni_count)

    #print(tokenizer.convert_ids_to_tokens(ids[i]),"+", tokenizer.convert_ids_to_tokens(ids[i+1]), bi_count)
    #print(tokenizer.convert_ids_to_tokens(ids[i]), "has overall count:", uni_count)
    print("P(", tokenizer.convert_ids_to_tokens(ids[i]),"+", tokenizer.convert_ids_to_tokens(ids[i+1]), ") = ", P)
"""

#################################
## Evaluate every sentence in the test corpus
for sentence in test_corpus:
  print(sentence)
  for i in range(len(sentence)-1):
    tok0_vid = np.where(vocab == sentence[i])[0][0]
    tok1_vid = np.where(vocab == sentence[i+1])[0][0]

    bi_count = bigram["freqs"][tok0_vid][tok1_vid]
    uni_count = np.sum(bigram["freqs"][tok0_vid])

    alpha = 0.009 # We must tune this from the dev set!
    P = (bi_count + alpha) / (uni_count + alpha * vocab.size)
    #P = (bi_count) / (uni_count)

    #print(tokenizer.convert_ids_to_tokens(ids[i]),"+", tokenizer.convert_ids_to_tokens(ids[i+1]), bi_count)
    #print(tokenizer.convert_ids_to_tokens(ids[i]), "has overall count:", uni_count)
    print("P(", tokenizer.convert_ids_to_tokens(sentence[i]),"+", tokenizer.convert_ids_to_tokens(sentence[i+1]), ") = ", P)


