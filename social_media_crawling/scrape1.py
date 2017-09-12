import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from twython import Twython
from datetime import datetime
import re
import os
import sqlite3

def has_badword(tweet):
    badword = ['bokep', 'ngentot', 'bisyar', 'mesum','coli', 'toket', 'ngewe','subscirbe','like', 'comment']
    for word in badword:
        if word in tweet:
            return True
        
def content_only(word, keyword):
    if keyword in word:
        return True
    
def month1(month):
    list1 = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'des']
    b = 0
    for a in list1:
        b += 1
        if(month == a):
            return str(b)
def bulan(bulan):
    list2 = ['jan', 'feb','mar','apr','mei','jun','jul','agu','sep','okt','nov','des']
    b = 0
    for a in list2:
        b += 1
        if(bulan == a):
            return str(b)

def bac(dat,lang,n):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_temp = os.path.join(BASE_DIR, 'db.sqlite3')

    
    conn = sqlite3.connect(db_temp)
    cursor = conn.cursor()
    
    browser = webdriver.Chrome()
    url = "https://twitter.com/search?src=typd&q="+str(dat)+"&l="+lang

    browser.get(url)
    time.sleep(1)
    
    body = browser.find_element_by_tag_name('body')
    
    for _ in range(n):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        
    
    html = browser.page_source
    soup = BeautifulSoup(html, "html.parser")
    browser.quit()
    b = soup.find_all("div", class_="content")
    indoTime=['jam','mnt','dtk']

    for a in b:
        #username
        name = a.find("strong", class_="fullname")
        name = name.text
        #tweets
        tweets = a.find("p", class_="TweetTextSize")
        tweet = tweets.text
        tweet = re.sub(r'http://\S+',"", tweet, re.IGNORECASE)
        tweet = re.sub(r'https://\S+',"", tweet, re.IGNORECASE)
        tweet = re.sub(r'pic.\S+',"", tweet, re.IGNORECASE)
        tweet = re.sub('[$<>:%&;]','', tweet)
        #date
        date = a.find("span", class_="_timestamp")
        date = date.text
        if lang =='in':
            m = date.split()
            if content_only(indoTime, m[1]):
                date = time.strftime("%Y-%m-%d")
            elif len(date)<=6:
                date = time.strftime("%Y ")+date
                m = date.split()
                m[2] = bulan(m[2].lower())
                temp = m[2]
                m[2] = m[1]
                m[1] = temp
                date = '-'.join(m)
            else:
                m = date.split()
                m[2] = bulan(m[2].lower())
                temp = m[2]
                m[2] = m[1]
                m[1] = temp
                date = '-'.join(m)
        else:        
            if len(date)<=3:
                date = time.strftime("%Y-%m-%d")
            elif len(date)<=6:
                date = time.strftime("%Y ")+date
                m = date.split()   
                m[1]=month1(m[1].lower())
                date = '-'.join(m)
            else:
                m = date.split()
                m[1]=month1(m[1].lower())
                date = '-'.join(m)
        if has_badword(tweet.lower()):
            continue
        if content_only(tweet.lower(),str(dat).lower()):
            cursor.execute("INSERT INTO WMMS_Crawl(tanggal,content,name) VALUES (?, ?, ?);", (date,tweet,name))
            conn.commit()
           
        
    
def API(keyword, bah):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_temp = os.path.join(BASE_DIR, 'db.sqlite3')

    
    conn = sqlite3.connect(db_temp)
    cursor = conn.cursor()
    
    APP_KEY='vEynfRampHnHyYAhD1yNdI2mB'
    APP_SECRET='pT7gVKTLPjyGYLjK4YXglH1cPrVxcSGAA1wIrICsFkv4DV5KEw'
    OAUTH_TOKEN='795951804219473920-c1DK5kzQxgY15HzVr985JiJyhroDFv7'
    OAUTH_TOKEN_SECRET='E5QRfRQQ7pRI01bvYx8uDUI9ObpiNGf6aZbJIxzcyLbM4'
    
    twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN,OAUTH_TOKEN_SECRET)
    
    
    results = twitter.search(q=keyword,count = '100', result_type = 'recent', lang = bah , max_id = None)
    for result in results['statuses']:
        name = result['user']['screen_name']
        content = (result['text'])
        content = re.sub(r'http://[\w.]+/+[\w.]+',"", content, re.IGNORECASE)
        content = re.sub(r'https://[\w.]+/+[\w.]+',"", content, re.IGNORECASE)
        content = re.sub('[$<>:%;]','', content)
        c_at =(result['created_at'])
        c_at = datetime.strptime(c_at,'%a %b %d %H:%M:%S +0000 %Y').strftime('%Y-%m-%d')
        if has_badword(content.lower()):
            continue
        if content_only(content.lower(),keyword.lower()):
            cursor.execute("INSERT INTO WMMS_Crawl(c_at,content,name) VALUES (?, ?, ?);", (c_at,content,name))
            conn.commit()