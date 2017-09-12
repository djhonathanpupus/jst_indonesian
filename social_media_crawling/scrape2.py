import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import common
from bs4 import BeautifulSoup
from twython import Twython
from datetime import datetime
import re
import os
import sqlite3

def has_badword(tweet , badword):
    if badword == "" or badword == None:
        return False
    badword = badword.split(",")
#     badword = ['bokep', 'ngentot', 'bisyar', 'mesum','coli', 'toket', 'ngewe','subscirbe','like', 'comment']
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
# def bulan(bulan):
#     list2 = ['jan', 'feb','mar','apr','mei','jun','jul','agu','sep','okt','nov','des']
#     b = 0
#     for a in list2:
#         b += 1
#         if(bulan == a):
#             return str(b)

def cleantweet(text):
    text = re.sub(r'http://\S+',"", text, re.IGNORECASE)
    text = re.sub(r'https://\S+',"", text, re.IGNORECASE)
    text = re.sub(r'pic.\S+',"", text, re.IGNORECASE)
    text = re.sub('[$<>:%&;]','', text)
    return text

def splithashtags(a):
    c = []
    for word in a:
        if word[0]=='#':
            c.append(word)
    return c

def RTcheck(text):
    if text[0] == 'RT':
        return True

def splitRT(text):
    a = text[1][:-1]
    b = text[2:]
    b = " ".join(b)
    return b ,a

def scrapeTwitter(dat,lang,n,tanggal,badword, cleandata):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_temp = os.path.join(BASE_DIR, 'db.sqlite3')
    cleaned_data = True
    print(cleandata)
    
    conn = sqlite3.connect(db_temp)
    cursor = conn.cursor()
    
    
    if int(tanggal[0])==0:
        url = "https://twitter.com/search?src=typd&q="+str(dat)+"&l="+lang

    else:
        url ="https://twitter.com/search?src=typd&l="+lang+"&q="+str(dat)+"%20since%3A"+tanggal[1]+"%20until%3A"+tanggal[2]

    browser = webdriver.Chrome()
    browser.get(url)
    time.sleep(1)
    
    body = browser.find_element_by_tag_name('body')
    ## Fungsi Scroll down
    for _ in range(n):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        
    
    html = browser.page_source
    soup = BeautifulSoup(html, "html.parser")
#     browser.quit()
    b = soup.find_all("div", class_="content")
#     indoTime=['jam','mnt','dtk']

    for a in b:
        RT = []
        #username
        name = a.find("strong", class_="fullname")
        name = name.text
        #tweets&hashtagss
        tweets = a.find("p", class_="TweetTextSize")
        tweet = tweets.text
        if cleandata == "Yes":
            cleaned_data = False
            tweet = cleantweet(tweet)
            cleaned_data = content_only(tweet.lower(),str(dat).lower())
        hashtag = splithashtags(tweet.split())
        hashtag = " ".join(hashtag)
        #date
        date = a.find("span", class_="_timestamp")
        date = date.text
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
            
        #Retweet
        retweet = a.find("div", class_="QuoteTweet-innerContainer")
    
        if retweet != None:
            user = a.find('b', class_="QuoteTweet-fullname")
            RT.append(user.text)
            post = a.find('div', class_="QuoteTweet-text")
            post = post.text
            if cleandata == "Yes":
                post = cleantweet(post)
            RT.append(post)
            
        if has_badword(tweet.lower(), badword.lower()):
            continue
        if cleaned_data:
            if len(RT)==0:
                cursor.execute("INSERT INTO social_media_crawling_TwitterCrawl(date,tweet,name,hashtag) VALUES (?, ?, ?, ?);", (date,tweet,name,hashtag))
                conn.commit()
            else :
                cursor.execute("INSERT INTO social_media_crawling_TwitterCrawl(date,tweet,name,hashtag,Retweet_user) VALUES (?, ?, ?, ?, ?);", (date,tweet,name,hashtag,RT[0]))
                conn.commit()
    conn.close()   
        
    
def API(keyword, bah, n, badword, cleandata):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_temp = os.path.join(BASE_DIR, 'db.sqlite3')
    cleaned_data = True
    
    conn = sqlite3.connect(db_temp)
    cursor = conn.cursor()
    
    APP_KEY='vEynfRampHnHyYAhD1yNdI2mB'
    APP_SECRET='pT7gVKTLPjyGYLjK4YXglH1cPrVxcSGAA1wIrICsFkv4DV5KEw'
    OAUTH_TOKEN='795951804219473920-c1DK5kzQxgY15HzVr985JiJyhroDFv7'
    OAUTH_TOKEN_SECRET='E5QRfRQQ7pRI01bvYx8uDUI9ObpiNGf6aZbJIxzcyLbM4'
    
    twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN,OAUTH_TOKEN_SECRET)
    
    results = twitter.search(q=keyword,count = n, result_type = 'recent', lang = bah , max_id = None)
    
    for result in results['statuses']:
        name = result['user']['screen_name']
        tweet = result['text']
        if cleandata == "Yes":
            cleaned_data = False
            tweet = cleantweet(tweet)
            cleaned_data = content_only(tweet.lower(),keyword.lower())
        hashtag = splithashtags(tweet.split())        
        hashtag = " ".join(hashtag)
        date =(result['created_at'])
        date = datetime.strptime(date,'%a %b %d %H:%M:%S +0000 %Y').strftime('%Y-%m-%d')
        if has_badword(tweet.lower(), badword):
            continue
        if cleaned_data:
            if RTcheck(tweet.split())== True:
                tweet , RT = splitRT(tweet.split())
                cursor.execute("INSERT INTO social_media_crawling_TwitterCrawl(date,tweet,name,hashtag,Retweet_user) VALUES (?, ?, ?, ?,?);", (date,tweet,name,hashtag,RT))
                conn.commit()
            else:
                cursor.execute("INSERT INTO social_media_crawling_TwitterCrawl(date,tweet,name,hashtag) VALUES (?, ?, ?, ?);", (date,tweet,name,hashtag))
                conn.commit()
    conn.close()

#Facebook
def getValue(data):
    data = data.split()
    data = data[0]
    data = changetoInt(data[0])
    return data
    
def changetoInt(data):
    panjang =len(data)
    if data[panjang-1]=='K':
        data = float(data[:panjang-1])*1000
        return int(data)
    elif data[panjang-1]=='M':
        data = float(data[:panjang-1])*1000000
        return int(data)
    return data

def checkifint(data):
    int = ['1','2','3','4','5','6','7','8','9','0']
    data = data.split()
    ind = 0
    for word in data:
        if word[0] in int :
            break
        ind += 1
    return data[ind]

def scrapeFacebook(keyword,n):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_temp = os.path.join(BASE_DIR, 'db.sqlite3')
    
    user = "renaldsalendah"
    pwd = "regger"
    
    
    conn = sqlite3.connect(db_temp)
    cursor = conn.cursor()
    
    driver = webdriver.Chrome()
    
    
    #keyword = "badan pusat statistik"
     
    #Log In to facebook 
    driver.get("http://www.facebook.com")
    time.sleep(1)
    
    #email
    elem = driver.find_element_by_id("email")
    elem.send_keys(user)
    #password
    elem = driver.find_element_by_id("pass")
    elem.send_keys(pwd)
     
    elem.send_keys(Keys.RETURN)
    
    time.sleep(2)
    action = webdriver.common.action_chains.ActionChains(driver)
    action.click()
    time.sleep(3)
    action = webdriver.common.action_chains.ActionChains(driver)
    action.click()
    action.perform()
    time.sleep(1)
    action = webdriver.common.action_chains.ActionChains(driver)
    action.click()
    action.perform()

    action.perform() 
    
    time.sleep(2)
    elem = driver.find_element_by_class_name("_1frb")
    elem.send_keys(keyword)
    elem.send_keys(Keys.RETURN)
#     elem = driver.find_element_by_class_name("_4jy0")
#     elem.click()
    
    time.sleep(2)
    action = webdriver.common.action_chains.ActionChains(driver)
    action.click()
    action.perform()
    time.sleep(3)
    action.click()
    action.perform()
    time.sleep(1)
    
    driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 't')
    driver.get('https://www.facebook.com/search/posts/?q='+keyword)
    
    body = driver.find_element_by_tag_name('body')
    for _ in range(10):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        
    #click See All
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    a = soup.find_all("span" ,class_= '_5dw8')
    a = a[len(a)-1]

    driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 'w')
    driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 't')
    driver.get('https://www.facebook.com'+(a.find("a")['href']))
    
    time.sleep(2)
    action = webdriver.common.action_chains.ActionChains(driver)
    action.click()
    action.perform()
    time.sleep(3)
    action.click()
    action.perform()
    time.sleep(1)
    
    
    ##auto tapdown
    body = driver.find_element_by_tag_name('body')
    
    for _ in range(n):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
    

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    
    linkpost = []

    link= soup.find_all('a', class_='_3084')
    for a in link:
        url = a['href']
        linkpost.append(url)
    for a in linkpost:
        #open all link
        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 't')
        driver.get('https://www.facebook.com'+a)
        html = driver.page_source                   #read datasource
        soup = BeautifulSoup(html, "html.parser")
        userinfo = soup.find('div', 'fbUserContent')     #find tag
        if userinfo == None:
            userinfo = soup.find('div','fbUserPost')
        name =  userinfo.find('span', class_='fwb')
        name = name.text
        status = userinfo.find('div', class_="userContent")
        spanclass = status.find("span", class_="text_exposed_hide")
        status = status.text
        if spanclass != None:
            status = re.sub(spanclass.text, ' ', status)
        like = userinfo.find('span', class_='_4arz')
        if like != None:
            like = like.text
            like = checkifint(like)
            like = changetoInt(like)
        else:
            like = "0"
        share = userinfo.find('a', class_='UFIShareLink')
        if share != None:
            share = share.text
            share = getValue(share)
        else:
            share ="0"
        comment = userinfo.find('div',class_='_4bl7')
        if comment != None:
            comment = comment.text
            comment = getValue(comment)
        else:
            comment="0"
            
        cursor.execute("INSERT INTO social_media_crawling_FacebookCrawl(status,like,name,comment,share) VALUES (?, ?, ?, ?, ?);", (status,like,name,comment,share))
        conn.commit()
    
        #close link
        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 'w')
    driver.quit()
        