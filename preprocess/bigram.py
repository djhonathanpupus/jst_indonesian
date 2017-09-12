'''
Created on 14 Jul 2017

@author: USER
'''
#bigram untuk file sqlite, *SELESAI*
#bigram untuk file sqlite, *SELESAI*
import re
import nltk
import jellyfish as jf
import csv
from collections import Counter
from nltk import ngrams

def words(text): return re.findall(r'\w+', text.lower())
file = open(r"C:\Users\USER\eclipse-workspace\WMSS2\WMSS2\preprocess\ind_news_2012_10k\ind_news_2012_10K-sentences.txt", "r", encoding="utf-8-sig").read()
n = 2
file = re.sub('[\d\W]',' ',str(file))
bigrams = ngrams(file.lower().split(), n)
# file = open(r"C:\Users\USER\Desktop\big.txt", "r", encoding="utf-8-sig")
# file = re.sub('[^a-zA-Z0-9#@]+', ' ', str(file))
freq = nltk.FreqDist(bigrams) #computes freq of occurrence
fdist = freq.keys() # sorted according to freq

WORDS = Counter(bigrams)
WORDS1 = Counter(file.lower().split())
#wordfreq = []
#for w in WORDS1:
    #wordfreq.append(WORDS1)
    #wordfreq[w]
    #print(wordfreq[w])

#bayes
def P(line): #N=sum(WORDS.values())
    "Probability of `word`."
    words = line.split()
    w1 = words[0]
    w2 = words[1]
    #print(wordfreq[w1])
    return WORDS[(w1, w2)]/WORDS1[w1]

#edit distance
def bigram_corr(line): #function with input line(sentence)
    words = line.split() #split line into words
    for idx, (word1, word2) in enumerate(zip(words[:-1], words[1:])):
#     line = list(itertools.chain.from_iterable(line))
        for i,j in fdist: #iterate over bigrams
            if (word2==j) and (jf.levenshtein_distance(word1,i) < 5): #if 2nd words of both match, and 1st word is at an edit distance of 2 or 1, replace word with highest occurring bigram
                idx = 0
                words[idx] = i
            elif (word1==i) and (jf.levenshtein_distance(word2,j) < 5):
                idx = 1
                words[idx] = j
    return " ".join(words)

def candidate(line):
    return ([bigram_corr(line)] or [line])

def bigram_corr2(line):
    try:
        return max([bigram_corr(line)], key=P)
    except ZeroDivisionError:
        return line

def bigram_corr3(line): #satu kalimat
    line = line.strip()
    line = " ".join(line.split())
    n = 2
    bigrams1 = re.sub('[^a-zA-Z0-9#@]+', ' ', str(line))
#     bigrams1 = re.sub('[\d\W]',' ',line)
#     bigrams1 = [b for b in zip(line.lower().split(" ")[:-1], line.lower().split(" ")[1:])]
    bigrams1 = ngrams(bigrams1.lower().split(), n)
#     print(bigrams1)
    hasil=[]
    for index, line in enumerate(bigrams1):
        if index == 0:
            hasil = bigram_corr2(' '.join(line)).split(' ')
#             hasil = bigram_corr2(' '.join(line))
        else:
            hasil.append(bigram_corr2(' '.join(line)).split(' ')[1])
#             hasil.append(bigram_corr2(' '.join(line))[1])
    return ' '.join(hasil)
#     return hasil

def bigram_corr4(): #untuk file sqlite
    import sqlite3

    f = sqlite3.connect("C:/Users/USER/Desktop/Tingkat 4/Skripsi/final.sqlite")
    cursor = f.cursor()

    #create table
    #cursor.execute('''CREATE TABLE TWEETS
    #(ID INTEGER PRIMARY KEY AUTOINCREMENT,
    #NAME           TEXT    NOT NULL,
    #TWEET          TEXT    NOT NULL);''')

    #read
    a = cursor.execute("SELECT TWEET from TWEETS")
    tweet = []
    for row in a:
#     print(row)
        tweet.append(row[0])
#     hasil=[]
    #delete
    #a= cursor.execute("DELETE from TWEETS")
    #f.commit()
    
    f.close()
    
    hasil=[]
    for line in tweet:
#         line.rstrip()
        hasil.append(bigram_corr3(line))
#         print(hasil)
    #return ''.join(str(hasil))
    return hasil


#print(bigram_corr4()) #penyelesaian untuk kata yang tidak ada di corpus
def bigram_corr5(): #untuk file csv

    with open("C:/Users/USER/Desktop/Tingkat 4/Skripsi/TWEETS2.csv", encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        hasil=[]
        for row in reader:
            hasil.append(bigram_corr3(row[2]))
        return hasil

def analisis_bigram():
    
    with open("C:/Users/USER/Desktop/Tingkat 4/Skripsi/TWEETS_edit.csv", encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        hasil=[]
        for row in reader:
            hasil.append(bigram_corr3(row[0]))      
#     hasil2 = re.sub('[^a-zA-Z0-9#@]+',' ',str(hasil))
    hasil2 = " ".join(hasil).split()
#         return hasil2
    with open(r"C:\Users\USER\Desktop\output_B.csv", "w",encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter ='\n',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        writer.writerow(hasil2)

def F_BG(lines): #satu kalimat
    separate1=[]
    lines = re.sub('[^a-zA-Z0-9#@]+', ' ', lines)
    separate1.append(lines.lower().split())
    for line in separate1:
        hasil=[]
        for word in line:
            r = requests.get('http://kateglo.com/api.php?format=json&phrase=' + word)
            if r.headers['content-type'] == 'application/json':
                wjdata = r.json()
                cek = wjdata['kateglo']['info']
                for cek in wjdata:
                    if wjdata['kateglo']['info'] == 'cak':
                        hasil.append(wjdata['kateglo']['definition'][0]['def_text'].split(';')[0])
                    elif wjdata['kateglo']['info'] == 'cak, kp':
                        hasil.append(wjdata['kateglo']['definition'][0]['def_text'].split(';')[0])
                    else :
                        hasil.append(word)
            else :
                hasil.append(word)
        a = ' '.join(hasil)
    line = a.strip()
    line = " ".join(line.split())
    n = 2
    bigrams1 = re.sub('[^a-zA-Z0-9#@]+', ' ', str(line))
#     bigrams1 = re.sub('[\d\W]',' ',line)
#     bigrams1 = [b for b in zip(line.lower().split(" ")[:-1], line.lower().split(" ")[1:])]
    bigrams1 = ngrams(bigrams1.lower().split(), n)
#     print(bigrams1)
    hasil=[]
    for index, line in enumerate(bigrams1):
        if index == 0:
            hasil = bigram_corr2(' '.join(line)).split(' ')
#             hasil = bigram_corr2(' '.join(line))
        else:
            hasil.append(bigram_corr2(' '.join(line)).split(' ')[1])
#             hasil.append(bigram_corr2(' '.join(line))[1])
    return ' '.join(hasil)
# analisis_bigram() 
# bigram_corr5()
# for a in bigram_corr4():
#     print(a)
# bigram_corr3("perppu jokowi's")