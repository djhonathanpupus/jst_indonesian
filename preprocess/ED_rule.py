'''
Created on 14 Jul 2017

@author: USER
'''
#Cek typo per kata - ED + Bayes
#Cek typo per kata - ED + Bayes
#Cek typo per kata - ED + Bayes
import re
import csv
import json
import requests
from urllib.request import urlopen
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

# def words(text): return re.findall(r'\w+', text.lower())
# file = open(r"C:\Users\USER\Desktop\big.txt", "r", encoding="utf-8-sig") #jgn lupa corpus di bikin lower text; buat fungsi
file = open(r"C:\Users\djhonathanpupus\Desktop\JST JHO\jst\preprocess\ind_news_2012_10k\ind_news_2012_10K-sentences.txt", "r", encoding="utf-8-sig").read()
file = re.sub('[\d\W]',' ',str(file))
    
WORDS = Counter(file.lower().split())

def rules(word):
    pjg = len(word)
    typo=[]
    if pjg == 1:
        if re.search(r"([a-zA-Z])", word) :
            typo.append(word)
    elif pjg == 2:
        if re.search(r"([a-zA-Z]+[a-zA-Z])", word) :
            typo.append(word)
    elif pjg > 2 :
        vocal = re.search(r"([a]+[i]|[a]+[u]|[e]+[i]|[o]+[i]|[k]+[h]|[n]+[g]|[n]+[y]|[s]+[y])", word)
#         cons = re.search(r"([k]+[h]|[n]+[g]|[n]+[y]|[s]+[y])", word)
        if not vocal:
            typo.append(word)
#         else :
#             typo.append(word)
#     elif re.search(r"([a-zA-Z]+[a-zA-Z]+[a-zA-Z]+[a-zA-Z])", word) :
#             typo.append(word)
    return typo
    
def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): #untuk kata
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# def correction_2(lines): #untuk kalimat ED + Bayes
#     separate=[]
#     separate.append(lines.split())
#     for line in separate:
#         hasil=[]
#         for word in line:
#             hasil.append(correction(word))
#         return ' '.join(hasil)

def correction_3(lines): #untuk kalimat Rule, ED + Bayes
#     lines = lines.strip()
#     lines = " ".join(lines.split())
    lines = re.sub('[^a-zA-Z0-9#@]+', ' ', lines)
#     lines = re.sub('[\d\W]',' ',lines)
    separate=[]
    separate.append(lines.lower().split())
    for line in separate:
        hasil=[]
        for word in line:
#             rules(word)
            if word in rules(word):
                hasil.append(correction(word))
            else:
                hasil.append(word)
        return ' '.join(hasil)           
    
def bigram_corr5(): #untuk file sqlite
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
#     print(tweet)
    for line in tweet:
        hasil.append(correction_3(line))
    return hasil
#     return ' '.join(hasil)


def bigram_corr6(): #untuk file csv
    import csv

    with open("C:/Users/USER/Desktop/Tingkat 4/Skripsi/TWEETS2.csv", encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        hasil=[]
        for row in reader:
            hasil.append(correction_3(row[2]))
        return hasil

def analisis_EDR():
    
    with open("C:/Users/USER/Desktop/Tingkat 4/Skripsi/TWEETS_edit.csv", encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        hasil=[]
        for row in reader:
            hasil.append(correction_3(row[0]))      
#     hasil2 = re.sub('[^a-zA-Z0-9#@]+',' ',str(hasil))
    hasil2 = " ".join(hasil).split()
#         return hasil2
    with open(r"C:\Users\USER\Desktop\output_EDR.csv", "w",encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter ='\n',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        writer.writerow(hasil2)

def F_EDR(lines): #formalisasi kemudian typo
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
#     lines = re.sub('[^a-zA-Z0-9#@]+', ' ', lines)
#     lines = re.sub('[\d\W]',' ',lines)
    separate=[]
    separate.append(str(a).split())
    for line in separate:
        hasil=[]
        for word in line:
            if word in rules(word):
                hasil.append(correction(word))
            else:
                hasil.append(word)
        return ' '.join(hasil) 

# analisis_EDR()
# for a in bigram_corr5():
#     print(a)
# print(correction_3('Saa mau percaya sama balairung1 #jokowi'))
# print(correction_2('jasaa menyebutka terdaka juga memperkayaa'))
# candidates('Kementriann')
# print(WORDS)
# rules('')