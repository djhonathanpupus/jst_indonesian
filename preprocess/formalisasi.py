import json
import codecs
import requests
import re
from urllib.request import urlopen


def rule(words):
    if words == "gw":
        return "gue"
    elif words == "lo":
        return "lu"
    
def correction(lines):
    separate = []
#     lines = re.sub('[\d\W]',' ',lines) #menghilangkan simbol + angka
    lines = re.sub('[^a-zA-Z0-9#@]+', ' ', lines)
    separate.append(lines.lower().split())
    for line in separate:
        hasil=[]
        for word in line:
            
            r = requests.get('http://kateglo.com/api.php?format=json&phrase=' + word)
            if r.headers['content-type'] == 'application/json':
#                 url = urlopen('http://kateglo.com/api.php?format=json&phrase=' + word).read()
#                 wjdata = json.loads(url.decode('utf-8'))
                wjdata = r.json()
                cek = wjdata['kateglo']['info']
                #return wjdata['kateglo']['definition']
                for cek in wjdata:
                    if wjdata['kateglo']['info'] == 'cak':
                        hasil.append(wjdata['kateglo']['definition'][0]['def_text'].split(';')[0])
                    elif wjdata['kateglo']['info'] == 'cak, kp':
                        hasil.append(wjdata['kateglo']['definition'][0]['def_text'].split(';')[0])
                    else:
                        hasil.append(word)
            else :
                hasil.append(word)
        return ' '.join(hasil)
                
def correction_2():
    import sqlite3

    f = sqlite3.connect("C:/Users/USER/Desktop/Tingkat 4/Skripsi/Test.sqlite")
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

    #delete
    #a= cursor.execute("DELETE from TWEETS")
    #f.commit()
    f.close()
    
    hasil=[]
    for line in tweet:
        hasil.append(correction(line))
    return hasil

def correction_csv(): #untuk file csv
    import csv

    with open("C:/Users/USER/Desktop/Tingkat 4/Skripsi/TWEETS2.csv", encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        hasil=[]
        for row in reader:
            hasil.append(correction_2(row[2]))
        return hasil