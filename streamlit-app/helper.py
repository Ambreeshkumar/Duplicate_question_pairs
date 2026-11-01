import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np
from gensim.models import Word2Vec
import scipy


# Word2Vec model load karna (Ensure karein ki file saved ho)
w2v_model = Word2Vec.load("../w2v_model.bin")


def test_common_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)


def test_total_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))


def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    
    # Stopwords ko pickle se load karna
    STOP_WORDS = pickle.load(open('../stopwords.pkl','rb'))

    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1]) if q1_tokens and q2_tokens else 0
    token_features[7] = int(q1_tokens[0] == q2_tokens[0]) if q1_tokens and q2_tokens else 0

    return token_features


def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0.0

    return length_features


def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4

    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


def preprocess(q):
    q = str(q).lower().strip()

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')

    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
        "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",
        "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
        "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
        "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not",
        "it's": "it is", "let's": "let us", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "should've": "should have", "shouldn't": "should not", "that's": "that is",
        "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
        "they've": "they have", "wasn't": "was not", "we'd": "we would", "we're": "we are",
        "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are",
        "what's": "what is", "when's": "when is", "where's": "where is", "who's": "who is",
        "why's": "why is", "won't": "will not", "would've": "would have", "wouldn't": "would not",
        "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"
    }

    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    q = BeautifulSoup(q, "html.parser").get_text()
    pattern = re.compile(r'\W')
    q = re.sub(pattern, ' ', q).strip()
    q = re.sub(r'\s+', ' ', q).strip()

    return q


def question_to_vector(question, model):
    words = question.lower().split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def query_point_creator(q1, q2):
    input_query = []

    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # fetch basic features (7 features)
    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    common_w = test_common_words(q1, q2)
    total_w = test_total_words(q1, q2)
    input_query.append(common_w)
    input_query.append(total_w)
    input_query.append(round(common_w / total_w, 2) if total_w > 0 else 0.0)

    # fetch token features (8 features)
    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    # fetch length based features (3 features)
    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    # fetch fuzzy features (4 features)
    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)
    # Total len of input_query is now 22 features

    # Word2Vec vectors for q1 and q2 (Har ek question ka 100-dim vector banega)
    q1_vec = question_to_vector(q1, w2v_model)
    q2_vec = question_to_vector(q2, w2v_model)

    # Sabhi features ko horizontal stack karna (22 + 100 + 100 = 222 features total)
    return np.hstack((np.array(input_query).reshape(1, 22), q1_vec.reshape(1, -1), q2_vec.reshape(1, -1)))