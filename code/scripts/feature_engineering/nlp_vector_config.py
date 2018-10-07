'''
Count Vector Config Overview
1. define stopwords
2. define extent of lemmatization
3. define tokenizer function (wrapper function for stopwords, lemmatization)
4. define vector config (wrapper object for tokenizer function)
5. define vectorization and matrix transformation

Vector Config ToDo
Apply more regorous corpus preprocessing:
=> remove all non-english entries, experiement with ngram forms 
=> remove redundant tokens
'''

from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer

# initialize constants
lemmatizer = WordNetLemmatizer()


def define_sw():
    custom_stop_words = ['–', '\u2019', 'u', '\u201d', '\u201d.',
                         '\u201c', 'say', 'saying', 'sayings',
                         'says', 'us', 'un', '.\"', 'would',
                         'let', '.”', 'said', ',”', 'ax', 'max',
                         'b8f', 'g8v', 'a86', 'pl', '145', 'ld9', '0t',
                         '34u']
    return set(stopwords.words('english') + custom_stop_words)


def lemmatize(token, tag):
    tag = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }.get(tag[0], wordnet.NOUN)

    return lemmatizer.lemmatize(token, tag)


def cab_tokenizer(document):
    tokens = []
    sw = define_sw()
    punct = set(punctuation)

    # split the document into sentences
    for sent in sent_tokenize(document):
        # tokenize each sentence
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            # preprocess and remove unnecessary characters
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # If punctuation, ignore token and continue
            if all(char in punct for char in token):
                continue

            # If stopword, ignore token and continue
            if token in sw:
                continue

            # Lemmatize the token and add back to the token
            lemma = lemmatize(token, tag)

            # Append lemmatized token to list
            tokens.append(lemma)
    return tokens


def generate_vector(params):
    return CountVectorizer(tokenizer=cab_tokenizer, ngram_range=tuple(params['ngram_range']),
                           min_df=params['min_doc_frequency'], max_df=params['max_doc_frequency'])


def vectorize(tf_vectorizer, df):
    # fit count vectorizer to supplied corpus, return term frequency matrix
    df = df.reindex(columns=['tweet'])  # reindex on tweet

    tf_matrix = tf_vectorizer.fit_transform(df['tweet'])
    tf_feature_names = tf_vectorizer.get_feature_names()

    return tf_matrix, tf_feature_names
