from django.test import TestCase
import os
import sqlite3
# Create your tests here.
def getall():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_temp = os.path.join(BASE_DIR, 'db.sqlite3')
    
    conn = sqlite3.connect(db_temp)
    cursor = conn.cursor()
    a = cursor.execute("SELECT NAME, TWEET, RETWEET_USER, HASHTAG from social_media_crawling_TwitterCrawl")
    c = {'users':[],'twits':[],'ritwistUsr':[],'ritwits':[],'hashtags':[]}
    for row in a:
        c['users'].append(row[0])
        c['twits'].append(row[1])
        c['ritwits'].append(row[2])
        c['hashtags'].append(row[3])
    return c

