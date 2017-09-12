from twython import Twython
from urllib import parse
APP_KEY='vEynfRampHnHyYAhD1yNdI2mB'
APP_SECRET='pT7gVKTLPjyGYLjK4YXglH1cPrVxcSGAA1wIrICsFkv4DV5KEw'
OAUTH_TOKEN='795951804219473920-c1DK5kzQxgY15HzVr985JiJyhroDFv7'
OAUTH_TOKEN_SECRET='E5QRfRQQ7pRI01bvYx8uDUI9ObpiNGf6aZbJIxzcyLbM4'

twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN,OAUTH_TOKEN_SECRET)

def getwoeid():
    results = twitter.get_available_trends()
    count = dict()
    CA = dict()
    for result in results:
        country = result['country']
        woeid = result['woeid']
        name = result['name']
        CC = result['countryCode']
        if country in count:
            b = count[country]
            b.append((name,woeid))
            count[country]=b
        if country not in count:
            count[country]=[(name,woeid)]
            CA[country]=CC
    return CA, count

def trending(a):
    results = twitter.get_place_trends(id=a)
    keyword ={}
    for result in results:
        for b in result['trends']:
            query = b['query']
            volume = b['tweet_volume']
            if volume!=None:
                keyword[query]= volume
    return keyword

def getCID(country):
    BS = getwoeid()
    length= len(BS[1][country])
    return BS[1][country][length-1][1]

def getTrending(country):
    if country == 'Worldwide':
        country = ''
    woeid = getCID(country)
    temps = trending(woeid)
    trend = []
    temps = sorted(temps, key=temps.__getitem__, reverse = True)
    for i in temps:
        b = parse.unquote(i)
        trend.append(b)
    return trend

def getAllTrending(country):
    #country = getCountry() #hapus comment jika udah siap pake
    trendss = {}
    for i in country:
        if i == 'Worldwide':
            i = ''
        woeid = getCID(i)
        temps = trending(woeid)
        trend = []
        temps = sorted(temps, key=temps.__getitem__, reverse = True)
        for j in temps[:5]:
            b = parse.unquote(j)
            trend.append(b)
        if i == '':
            i='Worldwide'
        trendss[i]=trend
    return trendss


def getCountry():
    BS = getwoeid()
    country = []
    for i in BS[0]:
        if i == '':
            i = 'Worldwide'
        country.append(i)
    return country