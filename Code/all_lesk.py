from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize
from nltk.util import ngrams
from nltk import pos_tag
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet as wn
import nltk
from udpy import UrbanClient
import pandas as pd
import string
from abc import ABC, abstractmethod
from itertools import chain
import pickle
import gensim
import gensim.downloader
import numpy as np
from scipy import spatial
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex

word2vec = gensim.downloader.load('word2vec-google-news-300')
termsim_index = WordEmbeddingSimilarityIndex(word2vec)
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


class Sense(ABC):
    def __init__(self, word, definition, examples):
        self.word = word
        self.definition = definition
        self.examples = examples
    
    @abstractmethod
    def get_context(self):
        pass
    
    def get_source(self):
        return type(self)
    
class UrbanDictionarySense(Sense):
    def __init__(self, word, definition, examples, upvotes, downvotes):
        super().__init__(word, definition, examples)
        self.upvotes = upvotes
        self.downvotes = downvotes
        
    def get_context(self, add_examples=True, augment_with_synonyms=True):
        tokens = preprocess_and_tokenize(self.definition)
        if add_examples:
            tokens = tokens + preprocess_and_tokenize(self.examples)
        if augment_with_synonyms:
            tokens = augment_tokens_with_synonyms(tokens)
        return tokens
    
class WordNetSense(Sense):
    def __init__(self, word, definition, examples):
        super().__init__(word, definition, examples)
        
    
    def get_context(self, add_examples=True, augment_with_synonyms=True):
        tokens = preprocess_and_tokenize(self.definition)
        if add_examples:
            tokens = tokens + preprocess_and_tokenize(self.examples)
        if augment_with_synonyms:
            tokens = augment_tokens_with_synonyms(tokens)
        return tokens


def get_synonyms(word):
    synonyms = wn.synsets(word)
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))

    return list(lemmas)


def augment_tokens_with_synonyms(tokens):
    synonyms = []
    for word in tokens:
        synonyms += get_synonyms(word)
        
    return list(set(list(tokens) + synonyms))


def get_senses_from_urban_dictionary_api(word):
    client = UrbanClient()

    try:
        results = client.get_definition(word)
    except NameError:
        # fails when input is invalid
        results = []
    return results

def get_senses_from_urban_dictionary(word):
    results = get_senses_from_urban_dictionary_api(word)
    
    # filtering only if word and sense match
    is_sense_match = lambda word,sense: word.lower() == sense.word.lower()
    results = [res for res in results if is_sense_match(word, res)]
    
    senses = [UrbanDictionarySense(res.word, res.definition, res.example, res.upvotes, res.downvotes) 
              for res in results]
    
    # sorting by diff in upvotes - downvotes
    senses.sort(key=lambda x: (x.upvotes-x.downvotes), reverse=True)
    
#     for sense in senses:
#         print (sense.word, "|", sense.definition)
    return senses[:20]


semcor_examples, semcor_sense_dict = None, None
with open("semcor_sense_examples.pickle", "rb") as f:
    semcor_examples, semcor_sense_dict = pickle.load(f)
    
def get_semcor_examples_for_word(word):
        examples = []
        for example_index in semcor_sense_dict.get(word, []):
            examples.append(semcor_examples[example_index])
        return " ".join(examples)
    
def get_senses_from_wordnet(word):
    synsets = wn.synsets(word)
    wordnet_senses = []
    
    for synset in synsets:
        wordnet_sense = WordNetSense(synset._name, synset.definition(), get_semcor_examples_for_word(synset._name))
        wordnet_senses.append(wordnet_sense)
    return wordnet_senses

def get_senses(word):
    return get_senses_from_urban_dictionary(word) + get_senses_from_wordnet(word)

def preprocess_and_tokenize(s):
    s = s.replace("[", "")
    s = s.replace("]", "")
    s=s.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    s = s.lower().split() #split into list
    s = [w for w in s if not w in nltk_stopwords.words("english")] #remove stopwords 
    return s


def get_context_for_mention(mention_sentence, mention, augment_with_synonyms=False):
#     stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
#     stopwords.append(mention)
    tokens = preprocess_and_tokenize(mention_sentence)
    if augment_with_synonyms:
        tokens = augment_tokens_with_synonyms(tokens)
    
    return tokens

def get_context_for_sense(sense):
    return sense.get_context(augment_with_synonyms=False, add_examples=True)
    
    
def lesk(sentence, target, verbose=False):
    #choose sense with most word overlap between definition in urban dict and context (not counting stop words)
    mention_context = get_context_for_mention(sentence, target)

    max_overlaps = 0
    max_wordnet_overlaps = 0
    max_ud_overlaps = 0
    overlaps_for_max = []
    max_sense_idx = -1
    lesk_sense = None

    senses = get_senses(target)
    best_wordnet_sense = None
    best_ud_sense = None
    for idx,sense in enumerate(senses):
        
        
        sense_context = get_context_for_sense(sense)
        
        #add matching by stem?
        overlaps = set(sense_context).intersection(mention_context)
        if verbose:
            print("_"*20)
            print(sense.word, sense.get_source().__name__)
            print(sense.definition)
            print("context", sense_context)
            print("OVERLAPS", overlaps)
        
        if len(overlaps) > max_overlaps:
            lesk_sense = sense
            max_overlaps = len(overlaps)
            overlaps_for_max = overlaps
            max_sense_idx = idx
        
        if sense.get_source() == WordNetSense:
            if not best_wordnet_sense or len(overlaps) > max_wordnet_overlaps:
                best_wordnet_sense = sense
                max_wordnet_overlaps = len(overlaps)

        if sense.get_source() == UrbanDictionarySense:
            if not best_ud_sense or len(overlaps) > max_ud_overlaps:
                best_ud_sense = sense
                max_ud_overlaps = len(overlaps)

    if max_wordnet_overlaps == max_ud_overlaps:
        lesk_sense = best_wordnet_sense
        max_overlaps = max_wordnet_overlaps
#     print(max_sense_idx, overlaps_for_max)
    #baseline with threshold=? is highest vote definition
    return lesk_sense, max_overlaps, 0, best_ud_sense, max_ud_overlaps, 0, best_wordnet_sense, max_wordnet_overlaps, 0
    


# answer,mx = lesk("About 4-5 years ago I sat at a bar here in Bangkok and argued with an \
#     English mate about Bitcoin. I’m an engineer and mathematician and had studied it \
#     extensively.We argued for hours. To prove my point, I said I would buy one Bitcoin \
#     and happily lose it when the system collapsed to prove my point. So I bought one BTC \
#     for around US$4,000.We argued for years.I sold it a few months ago for $48,000.\
#     Best investment I ever made!!!Ha ha. I wish I’d bought more.", "engineer")
# print ("Word:", "engineer")
# print ("Sense:", answer.word)
# print ("Source:", answer.get_source().__name__)
# print ("Definition:",answer.definition[:100])
# print ("Maximum overlaps:",mx)

def lesk_soft_cosine(sentence, target):
    #choose sense with most word overlap between definition in urban dict and context (not counting stop words)
    mention_context = get_context_for_mention(sentence, target)

    overlaps_for_max = []
    max_sense_idx = -1

    max_overlaps = 0
    best_sense = None
    max_sim = 0

    best_wordnet_sense = None
    max_wordnet_overlaps = 0
    max_wordnet_sim = 0

    best_ud_sense = None
    max_ud_overlaps = 0
    max_ud_sim = 0

    senses = get_senses(target)
    documents = [mention_context]
    for idx,sense in enumerate(senses):
        documents.append(get_context_for_sense(sense))
    dictionary = Dictionary(documents)
    documents = [ dictionary.doc2bow(x) for x in documents ]
    tfidf = TfidfModel(documents)      
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
    for idx,sense in enumerate(senses):

        sense_context = get_context_for_sense(sense)

        similarity = termsim_matrix.inner_product(documents[0], documents[idx +1], normalized=(True, True))
        print('similarity = %.4f' % similarity, sense.definition)       
        
        
        overlaps = set(sense_context).intersection(mention_context)

        
        if similarity > max_sim:
            best_sense = sense
            max_overlaps = len(overlaps)
            overlaps_for_max = overlaps
            max_sense_idx = idx
            max_sim = similarity

        if sense.get_source() == WordNetSense:
            if not best_wordnet_sense or similarity >= max_wordnet_sim:
                best_wordnet_sense = sense
                max_wordnet_overlaps = len(overlaps)
                max_wordnet_sim = similarity

        if sense.get_source() == UrbanDictionarySense:
            if not best_ud_sense or similarity >= max_ud_sim:
                best_ud_sense = sense
                max_ud_overlaps = len(overlaps)
                max_ud_sim = similarity


    if np.abs(max_wordnet_sim - max_ud_sim) < 0.00001 and best_wordnet_sense:
        best_sense = best_wordnet_sense
        max_overlaps = max_wordnet_overlaps
        max_sim = max_wordnet_sim


    if max_sim == 0:
        return lesk(sentence, target)

    return best_sense, max_overlaps, max_sim, best_ud_sense, max_ud_overlaps, max_ud_sim, best_wordnet_sense, max_wordnet_overlaps, max_wordnet_sim

# answer,mx,sim,ud_answer,ud_mx,ud_sim,wn_answer,wn_mx,wn_sim  = lesk_soft_cosine("About 4-5 years ago I sat at a bar here in Bangkok and argued with an \
#     English mate about Bitcoin. I’m an engineer and mathematician and had studied it \
#     extensively.We argued for hours. To prove my point, I said I would buy one Bitcoin \
#     and happily lose it when the system collapsed to prove my point. So I bought one BTC \
#     for around US$4,000.We argued for years.I sold it a few months ago for $48,000.\
#     Best investment I ever made!!!Ha ha. I wish I’d bought more.", "BTC")
# print ("Word:", "BTC")
# print ("Sense:", answer.word)
# print ("Source:", answer.get_source().__name__)
# print ("Definition:",answer.definition[:100])
# print ("Maximum overlaps:",mx)
# print ("Maximum sim:",sim)



def lesk_avg_cosine(sentence, target):
    #choose sense with most word overlap between definition in urban dict and context (not counting stop words)
    mention_context = get_context_for_mention(sentence, target)

    overlaps_for_max = []
    max_sense_idx = -1
    best_sense = None
    max_overlaps = 0
    max_sim = 0

    best_wordnet_sense = None
    max_wordnet_overlaps = 0
    max_wordnet_sim = 0


    best_ud_sense = None
    max_ud_overlaps = 0
    max_ud_sim = 0

    senses = get_senses(target)
    
    emd_mention_context = []
    count = 0
    for word in mention_context:
        if word in word2vec:
            emd_mention_context.append(word2vec[word])
            count +=1
    avg_mention_context = np.mean(emd_mention_context, axis =0)
    for idx,sense in enumerate(senses):


        sense_context = get_context_for_sense(sense)
        emd_sense_context = []
        count =0
        for word in sense_context:
            if word in word2vec:
                emd_sense_context.append(word2vec[word])
                count +=1
        avg_sense_context = np.mean(emd_sense_context, axis =0)

        sim = 1 - spatial.distance.cosine(avg_mention_context, avg_sense_context)

        overlaps = set(sense_context).intersection(mention_context)

        if sim >= max_sim:
            best_sense = sense
            max_overlaps = len(overlaps)
            overlaps_for_max = overlaps
            max_sense_idx = idx
            max_sim = sim
    
        if sense.get_source() == WordNetSense:
            if not best_wordnet_sense or sim >= max_wordnet_sim:
                best_wordnet_sense = sense
                max_wordnet_overlaps = len(overlaps)
                max_wordnet_sim = sim

        if sense.get_source() == UrbanDictionarySense:
            if not best_ud_sense or sim >= max_ud_sim:
                best_ud_sense = sense
                max_ud_overlaps = len(overlaps)
                max_ud_sim = sim
    
    if np.abs(max_wordnet_sim - max_ud_sim) < 0.00001 and best_wordnet_sense:
        best_sense = best_wordnet_sense
        max_overlaps = max_wordnet_overlaps
        max_sim = max_wordnet_sim

    if max_sim == 0:
        return lesk(sentence, target)

    return best_sense, max_overlaps, max_sim, best_ud_sense, max_ud_overlaps, max_ud_sim, best_wordnet_sense, max_wordnet_overlaps, max_wordnet_sim


# answer,mx,sim,ud_answer,ud_mx,ud_sim,wn_answer,wn_mx,wn_sim  = lesk_avg_cosine("About 4-5 years ago I sat at a bar here in Bangkok and argued with an \
#     English mate about Bitcoin. I’m an engineer and mathematician and had studied it \
#     extensively.We argued for hours. To prove my point, I said I would buy one Bitcoin \
#     and happily lose it when the system collapsed to prove my point. So I bought one BTC \
#     for around US$4,000.We argued for years.I sold it a few months ago for $48,000.\
#     Best investment I ever made!!!Ha ha. I wish I’d bought more.", "BTC")
# print ("Word:", "BTC")
# print ("Sense:", answer.word)
# print ("Source:", answer.get_source().__name__)
# print ("Definition:",answer.definition[:100])
# print ("Maximum overlaps:",mx)
# print ("Maximum sim:",sim)