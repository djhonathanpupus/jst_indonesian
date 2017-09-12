from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import re
import os

#hilangkan tanda baca dan stopwords
def preprocess( sentence ):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    dir_path = os.path.join(os.path.dirname(__file__), 'dataset')
    stopwords = set(map(lambda x: x.strip('\n') , open(os.path.join(dir_path, 'stopwords.txt'), 'r').readlines()))
    filtered_words = filter(lambda token: token not in stopwords, tokens)
    return " ".join(filtered_words)

def stem(sentence):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(sentence)

def initialize_berita(judul, isi):
    return {'token_judul': WhitespaceTokenizer().tokenize(stem(preprocess(judul))),
            'isi': isi,
            'list_isi': sent_tokenize(isi),
            'token_isi': [WhitespaceTokenizer().tokenize(stem(preprocess(kalimat))) for kalimat in sent_tokenize(isi)]}