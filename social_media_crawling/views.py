from django.shortcuts import render, render_to_response
from . import scrape1, scrape2
from . import trends
from .models import *
from django.http.response import HttpResponseRedirect
from django.http import HttpResponse
from django.db.models import Max
from .form import upFile, fixbrowse, fixscrape
from .tests import todlnn, getPost, usedvar, handle_uploaded_file, getFile, getKeyJson, uploadtoarray, getall, gettweet,gettweet1, handle, downloadfile, downloadfile1, toJst,downloadprepros
import os
from . import urls
import sqlite3
import json, csv, string
import numpy as np
from _io import StringIO
from wsgiref.util import FileWrapper
from .static.img.graph.visualisation import *
from JST.models import *
from preprocess.form import PostForm
from preprocess.form2 import PostForm2
from preprocess.ED_rule import *
from preprocess.ED import *
# from dlnn.listFunction import *
# from dlnn.listFunction2 import *
# from dlnn.tests import *
# Create your views here.

def delet(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_temp = os.path.join(BASE_DIR, 'db.sqlite3')
    
    conn = sqlite3.connect(db_temp)
    cursor = conn.cursor()
    cursor.execute("DELETE from social_media_crawling_TwitterCrawl;")
    conn.commit()
    return HttpResponseRedirect('../search2')

def delet2(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_temp = os.path.join(BASE_DIR, 'db.sqlite3')
    
    conn = sqlite3.connect(db_temp)
    cursor = conn.cursor()
    cursor.execute("DELETE from social_media_crawling_FacebookCrawl;")
    conn.commit()
    return HttpResponseRedirect('../search2')

# def changemetod(mth):
#     BASE_DIR = os.path.dirname((__file__))
#     var = os.path.join(BASE_DIR, 'var.txt')
#     f = open(var,'w')
#     f.write(mth)
#     usedvar.method = mth
#     f.close()

def index(request):
    return render(request, 'WMMS/index.html')


# def crawl(request):
#     #variabel
#     m1 = request.session.get('Method')
#     m2 = request.session.get('Lang')
#     m3 = request.session.get('Tapdown')
#     if m1==None:
#         ad = usedvar.method
#     else:
#         ad = m1
#     form1 = advOpt(request.POST)
#     form = browse_id(request.POST)
#     form2 = option(request.POST)
#     form3 = ScrapeOpt(request.POST)
#     #trend
#     #countryL = trends.getCountry() #hapus commen jika siap pake
#     #bad = trends.getAllTrending()
#     
#     #trend temp
#     countryL = ['Worldwide','Indonesia','United Kingdom']
#     bad = trends.getAllTrending(countryL)
# 
#     #form
#     if request.method == 'POST':
#         if 'methods' in request.POST:              
#             if form1.is_valid():
#                 a = request.POST.get('methods')
#                 changemetod(a)
#                 request.session['Method']=form1.cleaned_data['methods']
#                 ad = form1.cleaned_data['methods']
#             if form2.is_valid():
#                 request.session['Lang']=form2.cleaned_data['language']
#                 m2 = form2.cleaned_data['language']
#             if form3.is_valid():
#                 request.session['Tapdown']=form3.cleaned_data['tapdown']
#                 m3 = form3.cleaned_data['tapdown']
#                 
#         if 'crawl' in request.POST:
#             if form.is_valid():
#                 data = request.POST.get('content')               
#                 if ad == '0':
#                     scrape1.API(data,m2)
#                 if ad == '1':
#                     scrape1.bac(data,m2,m3)
#                 
#     return  render(request, 'WMMS/crawl.html', {
#         'form': browse_id(), 'form1':advOpt(initial={'methods':ad}),'form2':option(initial={'language':m2}),'form3':ScrapeOpt(initial={'tapdown':m3}), 'country':countryL, 'm1':ad, 'test':bad, "f3":upFile
#     })

def crawl2(request):
    #variabel
    twitterdata = TwitterCrawl.objects.all()
    facebookdata = FacebookCrawl.objects.all()
    analisis = ['None','JST','DLNN']
    badword = request.session.get('badword')
    cleandata = request.session.get('clean')
    checked = ''
    if cleandata == 'Yes':
        checked= 'checked'

    #trend
    #countryL = trends.getCountry() #hapus commen jika siap pake
    #bad = trends.getAllTrending()
    
    #trend temp
#     countryL = ['Worldwide','Indonesia','United Kingdom']
#     bad = trends.getAllTrending(countryL)

    #form
    if request.method == 'POST':
        form = fixbrowse(request.POST)
        form1 = fixscrape(request.POST)
        form2 = upFile(request.POST, request.FILES)
        
        if 'API' in request.POST:              
            if form.is_valid():
                data = request.POST.get('content')
                bahasa = request.POST.get('language')
                jumlah = request.POST.get('jumlah')
                scrape2.API(data,bahasa, jumlah, badword, cleandata)
#                 an = request.POST.get('analisis')
                prepross = request.POST.get('prepross')
                
                if prepross == "Yes":
#                     if an == "None":
                    hasil = ''
                    input1 = gettweet()
                    
                    return render(request, "hasil_crawl.html", {'inputDB':input1, 'hasil':hasil, 'f':PostForm, 'f2':PostForm2,'database':'twitter'})
                    
                    
#                 if an != None:
#                     if an == 'JST':
#                         return render(request, "WMMS/hasil_crawl2.html",{'database':'twitter'})
#                 
                                           
        if 'Scrape' in request.POST:
            if form.is_valid():
                data = request.POST.get('content') 
                bahasa = request.POST.get('language')    
                since = request.POST.get('since')
                until = request.POST.get('until')  
                tapdown = request.POST.get('tapdown')                     
                date = request.POST.get('date')
                date = [date,since,until]
                scrape2.scrapeTwitter(data,bahasa,int(tapdown),date,badword, cleandata)
                prepross = request.POST.get('prepross')
#                 an = request.POST.get('analisis')
#                 if an != "None" and prepross== "Yes":
#                     if an == "JST":
#                         return render(request, "WMMS/hasil_crawl2.html",{'database':'twitter'})
#                     
#                 elif an == 'None' and prepross == "Yes":
                if prepross == "YES":
                    hasil = ''
                    input1 = gettweet()
                        
                    return render(request, "hasil_crawl.html", {'inputDB':input1, 'hasil':hasil, 'f':PostForm, 'f2':PostForm2,'database':'twitter'})
            else:
                print(form.errors)
                
        if 'FScrape' in request.POST:
            data = request.POST.get('content')
            tapdown = request.POST.get('tapdown')
            scrape2.scrapeFacebook(data,int(tapdown))
            prepross = request.POST.get('prepross')
#             an = request.POST.get('analisis')
            
#             if an != "None" and prepross== "Yes":
#                 print('as')
# #                 chAnalisis(an)
#             elif an == "None" and prepross == "Yes":
            if prepross == "Yes":
                hasil = ''
                input1 = getPost()
                    
                return render(request, "hasil_crawl.html", {'inputDB':input1, 'hasil':hasil, 'f':PostForm, 'f2':PostForm2,'database':'facebook'})
                    
        if 'file' in request.FILES:
            crawler = request.POST.get('dataCrawl')
            print(crawler)           
            if form2.is_valid():
                handle(request.FILES['file'],request.FILES['file'].name)
            else:
                print(form.errors)  
        
        if 'download' in request.POST:
            dllist = request.POST.getlist('dlFile')
            return downloadfile(dllist, twitterdata)
        
        if 'download1' in request.POST:
            dllist = request.POST.getlist('dlFile')
            return downloadfile1(dllist, facebookdata)
        
        if 'optioncrawl' in request.POST:
            request.session['badword']=request.POST.get('badword')
            request.session['clean']=request.POST.get('cleanopt')
            
        
        if 'PP' in request.POST:
            hasil = ''
            input1 = ''
            crawler = request.POST.get('dataCrawl')
            if crawler == "twitter":
                input1 = gettweet()
            elif crawler == "facebook":
                input1 = getPost()

                        
            return render(request, "hasil_crawl.html", {'inputDB':input1, 'hasil':hasil, 'f':PostForm, 'f2':PostForm2,'database':crawler})

        
        if 'JST' in request.POST:
            crawler = request.POST.get('dataCrawl')

            return render(request, "WMMS/hasil_crawl2.html",{'database':crawler})

        if 'DLNN' in request.POST:
            Topik, FeatX, kelasSentimen = todlnn()
            input1 = ''
            crawler = request.POST.get('dataCrawl')
            if crawler == "twitter":
                input1 = gettweet()
            elif crawler == "facebook":
                input1 = getPost()

            return render(request,"WMMS/hasil_crawl3.html",{'database':crawler,'inputDB':input1,'data':FeatX,'sent':predict, 'topik':Topik, 'kelasSentimen':kelasSentimen})
        ####### handle preprocessing #########
                
        if 'inputA' in request.POST:
            del request.session['hasilpr']
            hasilpr = []
            input1 = ''
            data2 = []
            hasil = ''
            crawler = request.POST.get('dataCrawl')
            if crawler == "twitter":
                input1 = gettweet()
            elif crawler == "facebook":
                input1 = getPost()
            input2 = ' '.join(input1)
            situs1 = request.POST.get('method', '')
            if situs1=='EDR':
                for a in input1:
                    b = F_EDR(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            elif situs1 == 'ED':
                for a in input1:
                    b = F_ED(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            elif situs1 == 'BG':
                for a in input1:
                    b = F_BG(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            situs2 = request.POST.get('method1', '')
            if situs2 == 'FR':
                for a in data2:
                    a[1] = correction(a[1])
                    hasilpr.append(a[1])
            request.session['hasilpr'] = hasilpr
            return render(request, "hasil_prepross.html", {'dicthasil':data2, 'name1':situs1, 'name2':situs2})
        
        if 'downloadPreprocess' in request.POST:

            hasilpr = request.session.get('hasilpr')
            b = downloadprepros(hasilpr)
            return b
        
        if 'toJSTAnalisis' in request.POST:
            return render(request, "WMMS/hasil_crawl2.html",{'database':'preprocess'})
        
        ####### HANDLE JST #########
        
        if 'inputB' in request.POST:
            crawler = request.POST.get('dataCrawl')
            if crawler == "twitter":
                input1 = gettweet()
            elif crawler == "facebook":
                input1 = getPost()
            elif crawler == "preprocess":
                input1= request.session.get('hasilpr')
            statusMI = request.POST['statusMI']
            stopwords = request.POST['stopwords']
            vocabSize = int(request.POST['vocabSize'])
            dictData, kata,indexDoc, w, kalimat, statusMI = toJst(statusMI, vocabSize, stopwords, input1)
    

            return render(request, 'JST/previewMI.html', {'dictData': dictData, 'kata': kata, 'jarak': range(0, w),
                                                     'kalimat': kalimat, 'lenCorpus' : indexDoc, 'name':"hasilCrawl" ,
                                                    'statusMI': statusMI})
        
    ######## handle DLNN #######
        if 'input' in request.POST:         
            Topik, FeatX, kelasSentimen = todlnn() 
            tabledata = ''
            prediction = ''
            data = ''
            IA = ''
            topik = ''
            tabledata = []
            input1 = []
            crawler = request.POST.get('dataCrawl')
            if crawler == "twitter":
                input1 = gettweet()
            elif crawler == "facebook":
                input1 = getPost()
            elif crawler == "preprocess":
                input1= request.session.get('hasilpr')
            topik = request.POST.get("topik")
            FE = request.POST.get("FE")
            loadfile = input1
            datatemp = []
            for i in loadfile:
                datatemp.append(i)
            for i in datatemp:
                prepros, prediction = predict(i,int(FE),topik) #memanggil fungsi predict dari listFunction
                tabledata.append({
                    'input': i,
                    'prepros': prepros,
                    'prediction': prediction,
                    'confirm': True
                    })
            data = json.dumps(tabledata) #memasukan ke tabel hasil
            return render(request, "index_dlnnFinal.html",{'selected_topic': topik,'data':FeatX,'sent':predict, 'IA':IA, 'hasil':data,'topik':Topik, 'kelasSentimen':kelasSentimen})

               
    return  render(request, 'WMMS/social_media_crawling.html', {
        'form': fixbrowse(), 'form1':fixscrape(initial={'date':'0','tapdown':'50'}), 'f3':upFile, 'Tweets':twitterdata, 'statuses':facebookdata,'analisis':analisis, 'badword':badword, 'check':checked  })
    
# def fileUpload(request):
#     d, e = getFile()
#     listda ={}
#     c =[]
#     data = request.POST.get('data_choice')
#     if data != None:
#         listda[data] = getKeyJson(data)
#         c = listda[data]
# 
#     if request.method == 'POST':
#         form = upFile(request.POST, request.FILES)
#         if form.is_valid():
#             handle_uploaded_file(request.FILES['file'],request.FILES['file'].name)
#         else:
#             print(form.errors)
#         if 'TOTM' in request.POST:
#             print('succes')
#             datatwit = getall()
#             
#     return render(request,'WMMS/test.html',{"f3":upFile, 'data' : c, 'test':d, 'test1':e, 'sta':data})
# 
# def analisis1(request):
#     Cfeature = ['TF','TF-IDF','BOW','BIGRAM']
#     algorthm = ['SVM','Deep learning', 'Naive bayes']
#     hasil = "Sentimen Positif"
#     
#     if request.method == 'POST':
#         form = upFile(request.POST, request.FILES)
#         if form.is_valid():
#             print(uploadtoarray(request.FILES['file']))
#             print(request.POST.get('Cfeature'))
#             print(request.POST.get('algorthm'))
#         else:
#             print(form.errors)
#         
#         
#     return render(request,'WMMS/test1.html',{"f3":upFile, 'test':Cfeature, 'test1':algorthm, 'hasil':hasil})
# 
# def download(request):
#     twitterdata = TwitterCrawl.objects.all()
#     f = StringIO()
#     writer = csv.writer(f)
# 
#     for row in twitterdata:
#         writer.writerow([row.name,row.tweet,row.date,row.Retweet_user,row.hashtag])
#     f.flush()
#     f.seek(0)
#     response = HttpResponse(FileWrapper(f), content_type='text/csv')
#     response['Content-Disposition'] = 'attachment; filename=Data_twitter.csv'
#     return response

def visualisasi2(request):
    usrfav =visual1()
  
    
    return render(request, 'WMMS/visualFacebook.html',{'user':usrfav})    

def visualisasi(request):

    userfav = visual()
    res = TwitterCrawl.objects.filter().aggregate(max_id=Max('pk'))
    res.get('max_id')
    return render(request, 'WMMS/visualTwitter.html',{'user':userfav})

def savetoDB(request):
    FBtabel= FacebookCrawl.objects.all()
    TWtabel = TwitterCrawl.objects.all()
    
    if request.method=="POST":
        dbpick = request.POST.get("dataCrawl")
        name = request.POST.get('name')
        if dbpick == "twitter":
            r = TwitterTopik(topik = name)
            r.save()
            data = []
            for a in TWtabel:
                q = TwitterDataset(name = a.name, tweet=a.tweet,date = a.date, Retweet_user=a.Retweet_user,hashtag=a.hashtag,topik=r)
                q.save()
                data.append(a.tweet)
            visualDS(name,data,dbpick)
        if dbpick == "facebook":
            r = FacebookTopik(topik = name)
            r.save()
            data = []
            for a in FBtabel:
                q = FacebookDataset(name = a.name, status=a.status,like = a.like, comment = a.comment, share=a.share, topik=r)
                q.save()
                data.append(a.status)
            visualDS(name,data,dbpick)
    return HttpResponseRedirect("../")

def fullDBTW(request):
    topik = TwitterTopik.objects.all()
    db = ''
    twitter =''
    
    if request.method == "POST":
        if 'data_choice' in request.POST:
            db = request.POST.get('data_choice')
            twitter = TwitterDataset.objects.filter(topik=db)

        elif 'delete' in request.POST:
            check = request.POST.getlist('check')
            db_choice = request.POST.get('db_choice') 
            for a in check:
                TwitterDataset.objects.get(id=a).delete()
            
            ## renew Wordcloud
            twitter = TwitterDataset.objects.filter(topik=db_choice)
            db2 = twitter[0].topik.topik
            data = [a.tweet for a in twitter ]
            visualDS(db2, data, 'twitter')
            
            
        elif 'deleteDS' in request.POST:
            db_choice = request.POST.get('db_choice')
            TwitterTopik.objects.get(id=db_choice).delete()
            
        elif 'visual' in request.POST:
            db_choice = request.POST.get('db_choice')
            twitter = TwitterDataset.objects.filter(topik=db_choice)
            visualisation = TwitterTopik.objects.get(id=db_choice).topik
            return render(request, 'WMMS/full_db_tw.html',{'choice':db_choice,'topik':topik,'Tweets':twitter, 'visualisation':'GO', 'dbpicked':visualisation})
            
        if 'download' in request.POST:
            dllist = request.POST.getlist('dlFile')
            db_choice = request.POST.get('db_choice')
            twitter = TwitterDataset.objects.filter(topik=db_choice)
            return downloadfile(dllist, twitter)
        
        
        if 'PP' in request.POST:
            hasil = ''
            input1 = []
            db_choice = request.POST.get('db_choice')
            input2 = TwitterDataset.objects.filter(topik=db_choice).values('tweet')
            for a in input2:
                input1.append(a['tweet'])

                        
            return render(request, "hasil_crawl.html", {'inputDB':input1, 'hasil':hasil, 'f':PostForm, 'f2':PostForm2,'database':db_choice})

        
        if 'JST' in request.POST:
            db_choice = request.POST.get('db_choice')

            return render(request, "WMMS/hasil_crawl2.html",{'database':db_choice})

        if 'DLNN' in request.POST:
            Topik, FeatX, kelasSentimen = todlnn()
            input1 = []
            db_choice = request.POST.get('db_choice')
            input2 = TwitterDataset.objects.filter(topik=db_choice).values('tweet')
            for a in input2:
                input1.append(a['tweet'])


            return render(request,"WMMS/hasil_crawl3.html",{'database':db_choice,'inputDB':input1,'data':FeatX,'sent':predict, 'topik':Topik, 'kelasSentimen':kelasSentimen})
        ####### handle preprocessing #########
                
        if 'inputA' in request.POST:
            del request.session['hasilpr']
            hasilpr = []
            input1 = []
            data2 = []
            hasil = ''
            crawler = request.POST.get('dataCrawl')
            input2 = TwitterDataset.objects.filter(topik=crawler).values('tweet')
            for a in input2:
                input1.append(a['tweet'])
            input2 = ' '.join(input1)
            situs1 = request.POST.get('method', '')
            if situs1=='EDR':
                for a in input1:
                    b = F_EDR(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            elif situs1 == 'ED':
                for a in input1:
                    b = F_ED(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            elif situs1 == 'BG':
                for a in input1:
                    b = F_BG(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            situs2 = request.POST.get('method1', '')
            if situs2 == 'FR':
                for a in data2:
                    a[1] = correction(a[1])
                    hasilpr.append(a[1])
            request.session['hasilpr'] = hasilpr
            return render(request, "hasil_prepross.html", {'dicthasil':data2, 'name1':situs1, 'name2':situs2})
        
        if 'downloadPreprocess' in request.POST:

            hasilpr = request.session.get('hasilpr')
            b = downloadprepros(hasilpr)
            return b
        
        if 'toJSTAnalisis' in request.POST:
            return render(request, "WMMS/hasil_crawl2.html",{'database':'preprocess'})
        
        ####### HANDLE JST #########
        
        if 'inputB' in request.POST:
            input1 = []
            crawler = request.POST.get('dataCrawl')
            if crawler != 'preprocess':
                input2 = TwitterDataset.objects.filter(topik=crawler).values('tweet')
                for a in input2:
                    input1.append(a['tweet'])
            else:
                input1 = request.session.get('hasilpr')
            statusMI = request.POST['statusMI']
            stopwords = request.POST['stopwords']
            vocabSize = int(request.POST['vocabSize'])
            dictData, kata,indexDoc, w, kalimat, statusMI = toJst(statusMI, vocabSize, stopwords, input1)
    

            return render(request, 'JST/previewMI.html', {'dictData': dictData, 'kata': kata, 'jarak': range(0, w),
                                                     'kalimat': kalimat, 'lenCorpus' : indexDoc, 'name':"hasilCrawl" ,
                                                    'statusMI': statusMI})
        
    ######## handle DLNN #######
        if 'input' in request.POST:         
            Topik, FeatX, kelasSentimen = todlnn() 
            tabledata = ''
            prediction = ''
            data = ''
            IA = ''
            topik = ''
            tabledata = []
            input1 = []
            crawler = request.POST.get('dataCrawl')
            if crawler != 'preprocess':
                input2 = TwitterDataset.objects.filter(topik=crawler).values('tweet')
                for a in input2:
                    input1.append(a['tweet'])
            else:
                input1 = request.session.get('hasilpr')
            topik = request.POST.get("topik")
            FE = request.POST.get("FE")
            loadfile = input1
            datatemp = []
            for i in loadfile:
                datatemp.append(i)
            for i in datatemp:
                prepros, prediction = predict(i,int(FE),topik) #memanggil fungsi predict dari listFunction
                tabledata.append({
                    'input': i,
                    'prepros': prepros,
                    'prediction': prediction,
                    'confirm': True
                    })
            data = json.dumps(tabledata) #memasukan ke tabel hasil
            return render(request, "index_dlnnFinal.html",{'selected_topic': topik,'data':FeatX,'sent':predict, 'IA':IA, 'hasil':data,'topik':Topik, 'kelasSentimen':kelasSentimen})
            
    return render(request, 'WMMS/full_db_tw.html',{'choice':db,'topik':topik,'Tweets':twitter})

def fullDBFB(request):
    topik = FacebookTopik.objects.all()
    statuses = ''
    db = ''
    if request.method == "POST":
        if 'data_choice' in request.POST:
            db = request.POST.get('data_choice')
            statuses = FacebookDataset.objects.filter(topik=db)

        elif 'delete' in request.POST:
            check = request.POST.getlist('check')
            db_choice = request.POST.get('db_choice')
            for a in check:
                FacebookDataset.objects.get(id=int(a)).delete()
                
            ## renew wordcloud 
            statuses = FacebookDataset.objects.filter(topik=db_choice)    
            db2 = statuses[0].topik.topik
            data = [a.status for a in statuses]
            visualDS(db2,data,'facebook')
            
        elif 'deleteDS' in request.POST:
            db_choice = request.POST.get('db_choice')
            FacebookTopik.objects.get(id=db_choice).delete()
        
        elif 'visual' in request.POST:
            db_choice = request.POST.get('db_choice')
            statuses = FacebookDataset.objects.filter(topik=db_choice)
            visualisation = FacebookTopik.objects.get(id=db_choice).topik
            return render(request, 'WMMS/full_db_fb.html',{'choice':db_choice,'topik':topik,'statuses':statuses, 'visualisation':'GO', 'dbpicked':visualisation})
        
        if 'PP' in request.POST:
            hasil = ''
            input1 = []
            db_choice = request.POST.get('db_choice')
            input2 = FacebookDataset.objects.filter(topik=db_choice).values('status')
            for a in input2:
                input1.append(a['status'])

                        
            return render(request, "hasil_crawl.html", {'inputDB':input1, 'hasil':hasil, 'f':PostForm, 'f2':PostForm2,'database':db_choice})

        if 'download1' in request.POST:
            dllist = request.POST.getlist('dlFile')
            db_choice = request.POST.get('db_choice')
            statuses = FacebookDataset.objects.filter(topik=db_choice)
            return downloadfile1(dllist, statuses)
        
        if 'JST' in request.POST:
            db_choice = request.POST.get('db_choice')

            return render(request, "WMMS/hasil_crawl2.html",{'database':db_choice})

        if 'DLNN' in request.POST:
            Topik, FeatX, kelasSentimen = todlnn()
            input1 = []
            db_choice = request.POST.get('db_choice')
            input2 = FacebookDataset.objects.filter(topik=db_choice).values('status')
            for a in input2:
                input1.append(a['status'])


            return render(request,"WMMS/hasil_crawl3.html",{'database':db_choice,'inputDB':input1,'data':FeatX,'sent':predict, 'topik':Topik, 'kelasSentimen':kelasSentimen})
        ####### handle preprocessing #########
                
        if 'inputA' in request.POST:
            del request.session['hasilpr']
            hasilpr = []
            input1 = []
            data2 = []
            hasil = ''
            crawler = request.POST.get('dataCrawl')
            input2 = FacebookDataset.objects.filter(topik=crawler).values('status')
            for a in input2:
                input1.append(a['status'])
            input2 = ' '.join(input1)
            situs1 = request.POST.get('method', '')
            if situs1=='EDR':
                for a in input1:
                    b = F_EDR(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            elif situs1 == 'ED':
                for a in input1:
                    b = F_ED(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            elif situs1 == 'BG':
                for a in input1:
                    b = F_BG(a)
                    hasilpr.append(b)
                    data2.append((a,b))
            situs2 = request.POST.get('method1', '')
            if situs2 == 'FR':
                for a in data2:
                    a[1] = correction(a[1])
                    hasilpr.append(a[1])
            request.session['hasilpr'] = hasilpr
            return render(request, "hasil_prepross.html", {'dicthasil':data2, 'name1':situs1, 'name2':situs2})
        
        if 'downloadPreprocess' in request.POST:

            hasilpr = request.session.get('hasilpr')
            b = downloadprepros(hasilpr)
            return b
        
        if 'toJSTAnalisis' in request.POST:
            return render(request, "WMMS/hasil_crawl2.html",{'database':'preprocess'})
        
        ####### HANDLE JST #########
        
        if 'inputB' in request.POST:
            input1 = []
            crawler = request.POST.get('dataCrawl')
            print(crawler)
            if crawler != 'preprocess':
                input2 = FacebookDataset.objects.filter(topik=crawler).values('status')
                for a in input2:
                    input1.append(a['status'])
            else:
                input1 = request.session.get('hasilpr')
            statusMI = request.POST['statusMI']
            stopwords = request.POST['stopwords']
            vocabSize = int(request.POST['vocabSize'])
            dictData, kata,indexDoc, w, kalimat, statusMI = toJst(statusMI, vocabSize, stopwords, input1)
    

            return render(request, 'JST/previewMI.html', {'dictData': dictData, 'kata': kata, 'jarak': range(0, w),
                                                     'kalimat': kalimat, 'lenCorpus' : indexDoc, 'name':"hasilCrawl" ,
                                                    'statusMI': statusMI})
        
    ######## handle DLNN #######
        if 'input' in request.POST:         
            Topik, FeatX, kelasSentimen = todlnn() 
            tabledata = ''
            prediction = ''
            data = ''
            IA = ''
            topik = ''
            tabledata = []
            input1 = []
            crawler = request.POST.get('dataCrawl')
            if crawler != 'preprocess':
                input2 = FacebookDataset.objects.filter(topik=crawler).values('status')
            for a in input2:
                input1.append(a['status'])
            else:
                input1 = request.session.get('hasilpr')
            topik = request.POST.get("topik")
            FE = request.POST.get("FE")
            loadfile = input1
            datatemp = []
            for i in loadfile:
                datatemp.append(i)
            for i in datatemp:
                prepros, prediction = predict(i,int(FE),topik) #memanggil fungsi predict dari listFunction
                tabledata.append({
                    'input': i,
                    'prepros': prepros,
                    'prediction': prediction,
                    'confirm': True
                    })
            data = json.dumps(tabledata) #memasukan ke tabel hasil
            return render(request, "index_dlnnFinal.html",{'selected_topic': topik,'data':FeatX,'sent':predict, 'IA':IA, 'hasil':data,'topik':Topik, 'kelasSentimen':kelasSentimen})
    return render(request, 'WMMS/full_db_fb.html',{'choice':db,'topik':topik,'statuses':statuses})

