from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from nltk.tokenize import WhitespaceTokenizer
from sklearn import metrics
from sklearn.svm import LinearSVC
import numpy as np
import pickle
import re
import os


model_path = os.path.join(os.path.dirname(__file__), 'model')
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')
vocab_path = os.path.join(model_path, 'vocab')

def regex_when(berita):
    when_word_list = set(map(lambda x: x.strip('\n') , open(os.path.join(dataset_path, 'kapan_list.txt'), 'r').readlines()))
    expression = re.compile(
        '(' +
        '|'.join(re.escape(item) for item in when_word_list) +
        '|\(\d+\/\d+\/\d+\)' +
        '|\(\d+\/\d+\)' +
        '|(\d{1,2}.\d{2})'
        ')')
    
    when_index = list()
    
    for index, kalimat in enumerate(berita):
        if expression.search(kalimat.lower()):
            when_index.append(index)
    
    return [berita[index] for index in when_index]

def f4_weight(list_sentences):
    f4 = list();
    for index, sentence in enumerate(list_sentences):
        other_sentences = [item for sublist in (list_sentences[:index]+list_sentences[index+1:]) for item in sublist]
        intercept = set(sentence).intersection(other_sentences)
        union = set(sentence).union(other_sentences)
        f4.append(len(intercept) / float(len(union)))
    
    return f4

def f5_weight( list_sentences, title ):
    f5 = list()
    for sentence in list_sentences:
        f5.append(len(set(title).intersection(sentence)) / float(len(set(title).union(sentence))))
    return f5

def f2_weight(list_sentences):
    f2 = list()
    corpus = pickle.load(open(os.path.join(model_path, 'corpus_token.p'), "rb" ))
    corpus_len = len(corpus)
    
    sentences_len = len(list_sentences)
    
    for sentence in list_sentences:
        #panjang setiap kalimat
        sentence_len = len(sentence)
        if(sentence_len == 0):
            continue
        f2_temp = 0
        for token in set(sentence):
            tfi = sentence.count(token)
            sentence_that_contain_word = len(list(filter(lambda sent: token in sent, list_sentences)))
            corpus_that_contain_word = len(list(filter(lambda sent: token in sent, corpus)))
            pkss = sentence_that_contain_word/len(list_sentences)
            pss = sentences_len/corpus_len
            if(corpus_that_contain_word == 0):
                continue
            else:
                pk = corpus_that_contain_word/corpus_len
                f2_temp += tfi*((pkss*pss)/pk)
        f2.append(f2_temp/sentence_len)
        
    return f2

def bow(list_sentences, vocab = 'all'):
    if vocab not in ['all', 'f4', 'f5', 'sum']:
        print('error')
        return
    
    vocab = pickle.load(open(os.path.join(vocab_path, vocab+"_bow_vocab.p"), "rb" ))
    count_vectorizer = CountVectorizer(analyzer = "word", vocabulary=vocab)
    return count_vectorizer.fit_transform([' '.join(sentence) for sentence in list_sentences]).toarray()

def tfidf(list_sentences, vocab = 'all'):
    if vocab not in ['all', 'f4', 'f5', 'sum']:
        print('error')
        return
    
    vocab = pickle.load(open(os.path.join(vocab_path, vocab+"_tfidf_vocab.p"), "rb" ))
    tfidf_vectorizer = TfidfVectorizer(analyzer = "word", vocabulary=vocab)
    return tfidf_vectorizer.fit_transform([' '.join(sentence) for sentence in list_sentences]).toarray()

def predict(berita, f2=list(), f4=list(), f5=list()):
    if(len(f2)<=0):
        f2=f2_weight(berita['token_isi'])
    if(len(f4)<=0):
        f4=f4_weight(berita['token_isi'])
    if(len(f5)<=0):
        f5=f5_weight(berita['token_isi'], berita['token_judul'])
    
    weight = list(map(lambda f2, f4, f5: [f2*30, f4*39, f5*49], f2, f4, f5))
    feature = list(map(lambda t, b, w: np.append(np.append(t, b), w), tfidf(berita['token_isi']), bow(berita['token_isi']), weight))
    clf = pickle.load(open(os.path.join(model_path, "all_btw_model.p"), "rb" ))
    code_prediction = clf.predict(feature)
    
    #labeling stop here
    label_prediction = list()
    for kalimat, prediction in zip(berita['list_isi'], code_prediction):
        label_prediction.append({'kalimat':  kalimat, 'kode': prediction})
        
    return label_prediction

def transform_output(judul, prediction_code):
    prediction = {
        'name': judul,
        'kiri': [
            {'name': 'apa', 'kiri': []},
            {'name': 'dimana', 'kiri': []},
            {'name': 'bagaimana', 'kiri': []}
        ],
        'kanan': [
            {'name': 'kapan', 'kanan': []},
            {'name': 'siapa', 'kanan': []},
            {'name': 'mengapa', 'kanan': []}
        ]
    }
    
    for index, p in enumerate(prediction_code):
        if(p['kode'][0]):
            prediction['kiri'][0]['kiri'].append({'name': p['kalimat']})
        if(p['kode'][1]):
            prediction['kiri'][1]['kiri'].append({'name': p['kalimat']})
        if(p['kode'][2]):
            prediction['kiri'][2]['kiri'].append({'name': p['kalimat']})
        if(p['kode'][3]):
            prediction['kanan'][0]['kanan'].append({'name': p['kalimat']})
        if(p['kode'][4]):
            prediction['kanan'][1]['kanan'].append({'name': p['kalimat']})
        if(p['kode'][5]):
            prediction['kanan'][2]['kanan'].append({'name': p['kalimat']})
    
    return prediction

def update_model(corpus_kalimat):
    kalimat_all = corpus_kalimat
    count_vectorizer = CountVectorizer(analyzer = "word", max_features = 5000)
    bow_train_data_features = count_vectorizer.fit_transform([clean.clean for clean in s])
    
    bow_vocabulary = count_vectorizer.vocabulary_
    pickle.dump(bow_vocabulary, open( 'model/vocab/' + t + '_bow_vocab.p', 'wb' ))
    
    tfidf_vectorizer = TfidfVectorizer(analyzer = "word")
    tfidf_train_data_features = tfidf_vectorizer.fit_transform([clean.clean for clean in s])
    
    tfidf_vocabulary = tfidf_vectorizer.vocabulary_
    pickle.dump(tfidf_v, open( 'model/vocab/' + t + '_bow_vocab.p', 'wb' ))
    
    # Numpy arrays are easy to work with, so convert the result to an array
    bow_train_data_features = bow_train_data_features.toarray()
    tfidf_train_data_features = tfidf_train_data_features.toarray()
    
    bow = bow_train_data_features
    bow_tfidf = list(map(lambda bow, tfidf: np.append(tfidf, bow), bow_train_data_features, tfidf_train_data_features))
    bow_tfidf_f = list(map(lambda data_, bow, tfidf: np.append(np.append(tfidf, bow), [data_.f2, data_.f4, data_.f5]), s, bow_train_data_features, tfidf_train_data_features))
    bow_weight = list(map(lambda data_, bow: np.append(bow, [data_.f2*30, data_.f4*39, data_.f5*49]), s, bow_train_data_features))
    bow_tfidf_weight = list(map(lambda data_, bow, tfidf: np.append(np.append(tfidf, bow), [data_.f2*30, data_.f4*39, data_.f5*49]), s, bow_train_data_features, tfidf_train_data_features))
    tfidf_f = list(map(lambda data_, tfidf: np.append(tfidf, [data_.f2, data_.f4, data_.f5]), s, tfidf_train_data_features))
    tfidf_weight = list(map(lambda data_, tfidf: np.append(tfidf, [data_.f2*30, data_.f4*39, data_.f5*49]), s, tfidf_train_data_features))
    
    Y = list(map(lambda data_: data_.tipe.split(', '), s))
    
    X = [bow, bow_tfidf, bow_tfidf_f, bow_weight, bow_tfidf_weight, tfidf_f, tfidf_weight]
    label = ['b', 'bt', 'btf', 'bw', 'btw', 'tf', 'tw']
    
    for x, l in zip(X, label):
        clf = OneVsRestClassifier(LinearSVC(random_state=0))
        y = MultiLabelBinarizer(classes=classes).fit_transform(Y)
        clf.fit(x, y)
        pickle.dump(clf, open( 'model/' + t + '_' + l + '_model.p', 'wb' ))
    
    return('done')

def get_kalimat():
    db = pw.SqliteDatabase('dataset/skripsi.db')

    class Kalimat(pw.Model):
        berita = pw.IntegerField()
        kalimat = pw.TextField()
        tipe = pw.CharField()
        clean = pw.TextField()
        f2 = pw.DoubleField()
        f4 = pw.DoubleField()
        f5 = pw.DoubleField()
        index_kalimat = pw.IntegerField()

        class Meta:
            database = db

    db.connect()
    
    kalimat = Kalimat.select().where((Kalimat.f2 > 0.0 or Kalimat.f4 > 0.0 or Kalimat.f5 > 0.0) and Kalimat.tipe != '')
    values = set(map(lambda k: k.berita, kalimat))
    kalimat_per_berita = [[y for y in kalimat if y.berita==index] for index in values]

    selected_all = [kalimat for berita in kalimat_per_berita for kalimat in berita]
    
    return selected_all