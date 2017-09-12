from django.shortcuts import render, redirect
from .models import SentimenDB, FormalisasiKataDB, KataFormalDB, StopwordsIDDB
import string, time, os, logging, csv, json, requests
from zipfile import ZipFile
import numpy as np
import scipy.special as scp
from io import StringIO, TextIOWrapper, BytesIO
from builtins import str
from django.http import HttpResponse
from wsgiref.util import FileWrapper
from collections import Counter
from preprocess.formalisasi import correction
from .jst import modelJST

# Create your views here.
def validasi(request):
    if request.method == 'POST':
        #Membuat lisr memuat kata2 MI Positive
        positiveMI = request.POST['positiveMI'].lower()
        positiveMIFormalisasi = positiveMI.split(',')

        #Membuat list yang memuat kata2 MI Negative
        negativeMI = request.POST['negativeMI'].lower()
        negativeMIFormalisasi = negativeMI.split(',')

        #Membuat list yang memuat dokumen2 dari corpus
        name = request.FILES['dataset'].name
        print("Nama file : "+str(name))
        typeFile = name.split('.')[1]
        if (typeFile == 'txt'):
            readers = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
        elif (typeFile == 'csv'):
            try:
                text = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
                readers = csv.reader(text)
            except:
                text = StringIO(request.FILES['dataset'].file.read().decode())
                readers = csv.reader(text)
        else:
            return render(request, 'JST/inputDataSentimen.html', {})
        
        arrData = []
        arrDataRaw = []

        for line in readers:
            kalimat = ''.join(line)
            kata = kalimat.split()
            if (len(kata) > 0):
                kata.remove(kata[0])
                kalimat = ' '.join(kata)
                # print("Teks : "+ kalimat)
                # print(" ")
                arrData.append(kalimat)

        # Memberikan nilai untuk hyperparameter dari User atau otomatis
        if (request.POST['alpha'] == ""):
            alpha = -1
        else:
            alpha = float(request.POST['alpha'])
        print("Nilai Alpha : "+str(alpha))

        if (request.POST['beta'] == ""):
            beta = -1
        else:
            beta = float(request.POST['beta'])
        print("Nilai Beta : "+str(beta))

        if (request.POST['gamma'] == ""):
            gamma = -1
        else:
            gamma = float(request.POST['gamma'])
        print("Nilai Gamma : "+str(gamma))

        if(request.POST['topics'] == ""):
            topics = 1
        else:
            topics = int(request.POST['topics'])
        print("Banyak Topik : "+str(topics))

        if(request.POST['iterasi'] == ""):
            iterasi = 1000
        else:
            iterasi = int(request.POST['iterasi'])
        print("Banyak Iterasi Gibbs : "+str(iterasi))

        # Cek stopwords
        stopwords = request.POST['stopwords']
        if (stopwords == 'yes'):
            statusStopwords = True
        else:
            statusStopwords = False
        print("Status stopwords : "+str(statusStopwords))

        statusLexicon = request.POST['FSL']
        if (statusLexicon == 'none'):
            statusFSL = False
            if (request.POST['filtered'] == ""):
                filtered = 0
            else:
                filtered = int(request.POST['filtered'])
        elif (statusLexicon == 'full'):
            statusFSL = True
            filtered = 0
        else:
            statusFSL = True
            if (request.POST['filtered'] == ""):
                filtered = 0
            else:
                filtered = int(request.POST['filtered'])
        print("Status FSL : "+str(statusFSL)+" "+str(filtered))

        # Mencari status dari file label untuk pengujian prior
        cekLabel = request.FILES.get('label', False)
        if (cekLabel != False):
            typeFile = (request.FILES['label'].name).split('.')[1]
            if (typeFile == 'txt'):
                labels = TextIOWrapper(request.FILES['label'].file, encoding='utf-8 ', errors='replace')
            elif (typeFile == 'csv'):
                try:
                    text = TextIOWrapper(request.FILES['label'].file, encoding='utf-8 ', errors='replace')
                    labels = csv.reader(text)
                except:
                    text = StringIO(request.FILES['dataset'].file.read().decode())
                    labels = csv.reader(text)
            else:
                return render(request, 'JST/inputDataMI.html', {})

            dictLabel = {}
            for key, label in enumerate(labels):
                label = int(''.join(label))
                dictLabel[key] = int(label)

            if (len(dictLabel) != len(arrData)):
                return render(request, 'JST/inputDataMI.html', {})

        #Lakukan Proses utuk MI
        simulationIteration = int(request.POST['iterasiSimulasi'])

        positiveTopics = []
        negativeTopics = []

        pi_dli = np.zeros((len(arrData), 2, simulationIteration))

        vocabSize = 0
        corpusSize = 0 #banyak kata dalam suatu corpus
        corpusLength = 0 #banyak dokumen dalam suatu corpus
        priorLabeled = 0
        aveDocSize = 0.0

        kataPositive = []
        kataNegative = []
        kalimatHasil = []

        waktuSimulasi = {}
        hyperparametersSimulasi = {}
        hyperparameter = ""
        dataSimulasi = []

        # Mengekstrak topic words untuk tiap label sentimen
        pengaliPeluang = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2500, 2000, 1500, 1200,
                          1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
        kali = 0

        for i in range(0, simulationIteration):
            logging.warning("SIMULASI KE - "+str(i+1))
            positiveSimulasi = {}
            negativeSimulasi = {}
            name = "JST MI Simulation : %s " % str(simulationIteration)
            jst = modelJST(alpha, beta, gamma, topics, name, statusStopwords, statusFSL, filtered, iterasi,
                           positiveMIFormalisasi, negativeMIFormalisasi)
            jst.execute_model(arrData)
            waktuSimulasi[i] = jst.processTime
            hyperparametersSimulasi[i] = str(round(jst.alpha, 4)) + " * " + \
                                         str(round(jst.beta, 4)) + " * " + \
                                         str(round(jst.gamma, 4))
            for d in range(0, jst.numDocs):
                for l in range(0, jst.rangeSentiLabs):
                    pi_dli[d][l][i] = jst.pi_dl[d][l]

            if(i == 0):
                kalimatHasil = jst.arrData
                hyperparameter = str(jst.alpha) + ", " + str(jst.beta) + "," + str(jst.gamma)
                vocabSize = jst.vocabSize
                corpusSize = jst.corpusSize
                priorLabeled = jst.labelPrior
                aveDocSize = jst.aveDocLength
                kataPositive = jst.pdataset.labeledPositiveWords
                kataNegative = jst.pdataset.labeledNegativeWords
                corpusLength = jst.numDocs


            # JSON untuk topik positive
            for z in range(0, topics):

                words_probs = {}
                for w in range(0, vocabSize):
                    words_probs[w] = [w, jst.phi_lzw[1][z][w]]
                topicsWords = sorted(words_probs.items(), key=lambda item: item[1][1], reverse=True)

                for pengali in pengaliPeluang:
                    if (topicsWords[0][1][1] * pengali < 90):
                        kali = pengali
                        break

                positiveTopic = []
                for i in range(0, 40):
                    positiveTopic.append([jst.id2word[topicsWords[i][1][0]], int(round(topicsWords[i][1][1] * kali))])
                positiveSimulasi[z] = positiveTopic
            positiveTopics.append(positiveSimulasi)

            # JSON untuk topik negative
            for z in range(0, topics):
                words_probs = {}
                for w in range(0, vocabSize):
                    words_probs[w] = [w, jst.phi_lzw[0][z][w]]
                topicsWords = sorted(words_probs.items(), key=lambda item: item[1][1], reverse=True)

                for pengali in pengaliPeluang:
                    if (topicsWords[0][1][1] * pengali < 90):
                        kali = pengali
                        break

                negativeTopic = []
                for i in range(0, 40):
                    negativeTopic.append([jst.id2word[topicsWords[i][1][0]], int(round(topicsWords[i][1][1] * kali))])
                negativeSimulasi[z] = negativeTopic
            negativeTopics.append(negativeSimulasi)


        #Membuat JSON unutk hasil peluang sentimen tiap dokumen dalam tiap simulasi
        for d in range(0, corpusLength):
            data = {}
            data['kalimat'] = kalimatHasil[d]
            for i in range(0, simulationIteration):
                data['positive_'+str(i)] = pi_dli[d][1][i]
                data['negative_'+str(i)] = pi_dli[d][0][i]
                if(pi_dli[d][1][i] > pi_dli[d][0][i]):
                    label = 1
                elif(pi_dli[d][1][i] < pi_dli[d][0][i]):
                    label = 2
                else:
                    label = 0
                data['hasil_'+str(i)] = label
            dataSimulasi.append(data)
        jsonSimulasi = json.dumps(dataSimulasi)

        #Membuat ringkasan label sentimen
        sentimenSimulasi = []
        for i in range(0, simulationIteration):
            sentimenLabel = []
            for d in range(0, corpusLength):
                if (pi_dli[d][1][i] > pi_dli[d][0][i]):
                    label = 'positive'
                elif (pi_dli[d][1][i] < pi_dli[d][0][i]):
                    label = 'negative'
                else:
                    label = 'netral'
                sentimenLabel.append(label)
                sentimenLabel.append('total')
            sentimenTest = Counter(sentimenLabel)
            sentimenSimulasi.append(sentimenTest)
        jsonSentimen = json.dumps(sentimenSimulasi)
        jsonPositive = json.dumps(positiveTopics)
        jsonNegative = json.dumps(negativeTopics)


        if(cekLabel == False):
            name = "JST MI tanpa label Simulation : %s " % str(simulationIteration)
            #Membuat json untuk review simulasi
            arrReviewSimulasi = []
            for i in range(0, simulationIteration):
                arrRStemp = {}
                arrRStemp['waktu'] = waktuSimulasi[i]
                arrRStemp['hyperparameter'] = hyperparametersSimulasi[i]
                arrRStemp['positive'] = round(
                    (sentimenSimulasi[i]['positive']/sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['negative'] = round(
                    (sentimenSimulasi[i]['negative'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['objektif'] = round(
                    (sentimenSimulasi[i]['netral'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrReviewSimulasi.append(arrRStemp)
            jsonReviewSimulasi = json.dumps(arrReviewSimulasi)

            return render(request, 'JST/HasilJSTSimulasi.html', {'corpusLength': corpusLength,
                                                                 'name': name,
                                                                 'stopwordsStatus': statusStopwords,
                                                                 'lexiconStatus': statusLexicon + " (" + str(
                                                                     filtered) + ")",
                                                                 'hyperparameters': hyperparameter,
                                                                 'vocabSize': vocabSize,
                                                                 'corpusSize': corpusSize,
                                                                 'aveDocSize': aveDocSize,
                                                                 'priorLabeled': priorLabeled,
                                                                 'topics': topics,
                                                                 'iterasiGibbs': iterasi,
                                                                 'kataPositive': kataPositive,
                                                                 'kataNegative': kataNegative,
                                                                 'jsonSimulasi': jsonSimulasi,
                                                                 'iterasiSimulasi': simulationIteration,
                                                                 'jsonPositive': jsonPositive,
                                                                 'jsonNegative': jsonNegative,
                                                                 'jsonSentimen': jsonSentimen,
                                                                 'jsonReviewSimulasi': jsonReviewSimulasi,
                                                                 })
        else:
            name = "JST MI dengan label Simulation : %s " % str(simulationIteration)
            #Membuat pengukuran terhadap akurasi
            akurasiSimulasi = {}
            for i in range(0, simulationIteration):
                sumDocLabel = 0
                sumDocAkurasi = 0
                for d in range(0, len(arrData)):
                    if (pi_dli[d][1][i] > pi_dli[d][0][i]):
                        sentiLab = 1
                        sumDocLabel += 1
                    elif (pi_dli[d][1][i] < pi_dli[d][0][i]):
                        sentiLab = 2
                        sumDocLabel += 1
                    else:
                        sentiLab = -1

                    if (str(sentiLab) == str(dictLabel[i])):
                        sumDocAkurasi += 1
                akurasiSimulasi[i] = round((sumDocAkurasi / sumDocLabel) * 100, 2)

            #membuat json untuk review simulasi dengan nilai akurasi labelnya
            arrReviewSimulasi = []
            for i in range(0, simulationIteration):
                arrRStemp = {}
                arrRStemp['waktu'] = waktuSimulasi[i]
                arrRStemp['hyperparameter'] = hyperparametersSimulasi[i]
                arrRStemp['positive'] = round(
                    (sentimenSimulasi[i]['positive'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['negative'] = round(
                    (sentimenSimulasi[i]['negative'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['objektif'] = round(
                    (sentimenSimulasi[i]['netral'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['akurasi'] = akurasiSimulasi[i]
                arrReviewSimulasi.append(arrRStemp)
            jsonReviewSimulasi = json.dumps(arrReviewSimulasi)
            return render(request, 'JST/HasilJSTSimulasi.html', {'corpusLength': corpusLength,
                                                                 'name': name,
                                                                 'stopwordsStatus': statusStopwords,
                                                                 'lexiconStatus': statusLexicon + " (" + str(
                                                                     filtered) + ")",
                                                                 'dictLabel': dictLabel,
                                                                 'hyperparameters': hyperparameter,
                                                                 'vocabSize': vocabSize,
                                                                 'corpusSize': corpusSize,
                                                                 'aveDocSize': aveDocSize,
                                                                 'priorLabeled': priorLabeled,
                                                                 'iterasiSimulasi': simulationIteration,
                                                                 'topics': topics,
                                                                 'iterasiGibbs': iterasi,
                                                                 'kataPositive': kataPositive,
                                                                 'kataNegative': kataNegative,
                                                                 'jsonSimulasi': jsonSimulasi,
                                                                 'jsonPositive': jsonPositive,
                                                                 'jsonNegative': jsonNegative,
                                                                 'jsonSentimen': jsonSentimen,
                                                                 'jsonReviewSimulasi': jsonReviewSimulasi,
                                                                 })




        return render(request, 'JST/halamanMuka.html', {})
    else:
        return render(request, 'JST/validasi.html', {})

def halamanMuka(request):
    return render(request, 'JST/halamanMuka.html', {})

def inputDataSentimen(request):
    return render(request, 'JST/inputDataSentimen.html', {})

def simpanSentimen(request):
    if request.method == 'POST':
        jenisFile = request.POST["jenisFile"]
        typeFile = (request.FILES['dataset'].name).split('.')[1]
        if (typeFile == 'txt'):
            readers = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
        elif (typeFile == 'csv'):
            try:
                text = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
                readers = csv.reader(text)
            except:
                text = StringIO(request.FILES['dataset'].file.read().decode())
                readers = csv.reader(text)
        else:
            return render(request, 'JST/inputDataSentimen.html', {})

        if (jenisFile == "positive"):
            sentimens = SentimenDB.objects.filter(sentiLab=1).values_list('kataSentimen', flat=True)
            for reader in readers:
                kata = ''.join(reader)
                #kata = str(reader)
                if kata not in sentimens:
                    priorPos = 0.90
                    priorNeg = 0.05
                    priorNet = 0.05
                    sentiLab = 1
                    sentimen = SentimenDB(kataSentimen=kata, sentiLab=sentiLab, priorPositive=priorPos,
                                          priorNegative=priorNeg, priorNetral=priorNet)
                    sentimen.save()
            return render(request, 'JST/halamanMuka.html', {})
        elif (jenisFile == "negative"):
            #SentimenDB.objects.all().delete()
            sentimenDict = SentimenDB.objects.filter(sentiLab=2).values_list('kataSentimen', flat=True)
            for reader in readers:
                kata = ''.join(reader)
                if kata not in sentimenDict:
                    priorPos = 0.05
                    priorNeg = 0.90
                    priorNet = 0.05
                    sentiLab = 2
                    sentimen = SentimenDB(kataSentimen=kata, sentiLab=sentiLab, priorPositive=priorPos,
                                          priorNegative=priorNeg, priorNetral=priorNet)
                    sentimen.save()
            return render(request, 'JST/halamanMuka.html', {})
        elif (jenisFile == "prior"):
            SentimenDB.objects.all().delete()
            sentimenDict = SentimenDB.objects.values_list('kataSentimen', flat=True)
            for reader in readers:
                kata = ''.join(reader)
                baris = kata.split()
                kata = baris[0]
                if kata not in sentimenDict:
                    priorNet = float(baris[1])
                    priorPos = float(baris[2])
                    priorNeg = float(baris[3])
                    if (priorPos > priorNeg):
                        if (priorPos > priorNet):
                            sentiLab = 1
                        elif (priorPos < priorNet):
                            sentiLab = -1
                    elif (priorPos < priorNeg):
                        if (priorNeg > priorNet):
                            sentiLab = 2
                        elif (priorNeg < priorNet):
                            sentiLab = -1
                    else:
                        sentiLab = -1

                    sentimen = SentimenDB(kataSentimen=kata, sentiLab=sentiLab, priorPositive=priorPos,
                                          priorNegative=priorNeg, priorNetral=priorNet)
                    sentimen.save()
            return render(request, 'JST/halamanMuka.html', {})

        elif (jenisFile == "sentilab"):
            return render(request, 'JST/inputDataSentimen.html', {})
        else:
            return render(request, 'JST/inputDataSentimen.html', {})
    else:
        return render(request,'JST/halamanMuka.html',{})

def simpanStopwords(request):
    if request.method == 'POST':
        # StopwordsIDDB.objects.all().delete()
        typeFile = (request.FILES['dataset'].name).split('.')[1]
        if (typeFile == 'txt'):
            readers = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
        elif (typeFile == 'csv'):
            try:
                text = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
                readers = csv.reader(text)
            except:
                text = StringIO(request.FILES['dataset'].file.read().decode())
                readers = csv.reader(text)
        else:
            return render(request, 'JST/inputDataSentimen.html', {})
        for line in readers:
            stopword = StopwordsIDDB(kataStopword=str(''.join(line)))
            stopword.save()
        #logging.warning("Save done")
        return render(request, 'JST/inputDataSentimen.html', {})

    else:
        return render(request, 'JST/inputDataSentimen.html', {})
    
def inputDataFormalisasi(request):
    if request.method == 'POST':
        #cek koneksi formalisasi
        koneksi = cekKoneksi()

        arrKataFormal = KataFormalDB.objects.values_list('kataFormal', flat=True)
        arrFormalisasi = FormalisasiKataDB.objects.values_list('kataInformal', flat=True)
        arrSentimen = SentimenDB.objects.values_list('kataSentimen', flat=True)

        arrData = []
        arrData.extend(arrKataFormal)
        arrData.extend(arrFormalisasi)
        arrData.extend(arrSentimen)
        arrData = list(set(arrData))
        dictKata = {}

        #file = request.FILES['dataset']
        #file.open()
        remove = string.punctuation
        remove = remove.replace("#","")
        remove = remove.replace("@","")

        #for line in file:
        #    line = str(line)
        #    line = line[2:-5]
        #    line = ''.join(line)
        typeFile = (request.FILES['dataset'].name).split('.')[1]
        if (typeFile == 'txt'):
            readers = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
        elif (typeFile == 'csv'):
            try:
                text = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
                readers = csv.reader(text)
            except:
                text = StringIO(request.FILES['dataset'].file.read().decode())
                readers = csv.reader(text)
        else:
            return render(request, 'JST/inputDataSentimen.html', {})
        dfjKata = {}
        numDocs = 0
        arrCorpus = []
        for reader in readers:
            kalimat = ''.join(reader)
            line = kalimat.translate(str.maketrans('', '', remove)).lower()
            arrCorpus.append(line)

        arrJSONFormalisasi = []
        #if(koneksi):
        #   for kalimat in arrCorpus:
        #        formalisasiKateglo = {}
        #        formalisasiKateglo['input'] = kalimat
        #        formalisasiKateglo['output'] = correction(kalimat)
        #        arrJSONFormalisasi.append(formalisasiKateglo)
        jsonFormalisasi = json.dumps(arrJSONFormalisasi)

        for line in arrCorpus:
            baris = line.split()
            if(len(baris) > 0):
                numDocs += 1
                #Untuk Unigram
                for kata in baris:
                    if kata not in arrData:
                        if kata not in dictKata.keys():
                            dictKata[kata] = 1
                            dfjKata[kata] = 0
                        else:
                            dictKata[kata] += 1
                # Untuk Bigram
                # for i in range(0, len(baris) - 1):
                #     kata = str(''.join(baris[i] + " " + baris[i+1]))
                #     if kata not in arrData:
                #         if kata not in dictKata.keys():
                #             dictKata[kata] = 1
                #             dfjKata[kata] = 0
                #         else:
                #             dictKata[kata] += 1

        # for reader in readers:
        #     kata = ''.join(reader)
        #     line = kata.translate(str.maketrans('', '', remove)).lower()
        #     baris = line.split()
        #     if(len(baris) > 0):
        #         for i in range(0, len(baris)-1):
        #             kata = baris[i] + " " + baris[i+1]
        #             if kata not in arrData:
        #                 if kata not in dictKata.keys():
        #                     dictKata[kata] = 1
        #                     dfjKata[kata] = 0
        #                 else:
        #                     dictKata[kata] += 1

        for reader in arrCorpus:
            baris = reader.split()
            # if(len(baris)>0):
            #     for i in range(0, len(baris) - 1):
            #         kata = str(''.join(baris[i] + " " + baris[i+1]))
            #         baris.append(kata)
            #Menghitung dfj
            for kata in dictKata.keys():
                if kata in baris:
                    if(dfjKata[kata] == 0):
                        dfjKata[kata] = 1
                    else:
                        dfjKata[kata] += 1

        #Inisialisasi dan hitung tf-idf
        tfidfKata = {}

        for kata in dictKata.keys():
            #logging.warning(kata)
            if(dfjKata[kata] == numDocs):
                n = 0
            else:
                n = 1
            tfidfKata[kata] = dictKata[kata] * np.log(numDocs/(dfjKata[kata]))
            #logging.warning(str(kata) +" : "+str(tfidfKata[kata]))

        #arrKata = sorted(dictKata, key=dictKata.__getitem__, reverse=True)
        arrKata = sorted(tfidfKata, key=tfidfKata.__getitem__, reverse=True)

        w = 0
        dictKata = {}
        for kata in arrKata:
            dictKata[w] = kata
            w += 1

        arrKataFormalizationed = []
        arrKataNonFormalizationed = []
        for kata in arrKata:
            data = {}
            data['input'] = kata
            data['output'] = correction(kata)
            if(data['input'] == data['output']):
                arrKataNonFormalizationed.append(kata)
            else:
                arrKataFormalizationed.append(data)
        #Catch error
        #if not arrKataFormalizationed:
        #    arrKataNonFormalizationed.append('Lala')
        jsonFormalized = json.dumps(arrKataFormalizationed)
        jsonNonFormalized = json.dumps(arrKataNonFormalizationed)

        batasKata = int(request.POST['vocabSize'])
        if(w > batasKata):
            w = batasKata

        return render(request, 'JST/formalisasiKata.html', {'dickKata': dictKata, 'arrData': arrData, 
                                                            'vocabSize': w,
                                                            'jsonFormalisasi': jsonFormalisasi,
                                                            'jsonFormalized': jsonFormalized,
                                                            'jsonNonFormalized': jsonNonFormalized,
                                                            })
    else:
        return render(request, 'JST/inputDataFormalisasi.html',{})

def simpanFormalisasiKata(request):
    if request.method == 'POST':
        vocabSize = int(request.POST['vocabSize'])
        for x in range(0,vocabSize):
            x = '_'+str(x)
            kataInformal = 'kataInformal'+x
            kataFormal = 'kataFormal'+x
            kataInformal = request.POST[kataInformal]
            kataFormal = request.POST[kataFormal]

            if (kataFormal != ""):
                form = FormalisasiKataDB(kataInformal=kataInformal, kataFormal=kataFormal)
                form.save()
            else:
                form = KataFormalDB(kataFormal=kataInformal)
                form.save()

        return redirect('JST:halamanMuka')
    else:
        return redirect('JST:inputData')

def previewMI(request):
    if request.POST:
        # arrKataFormal = KataFormalDB.objects.values_list('kataFormal', flat=True)
        # arrFormalisasi = FormalisasiKataDB.objects.values_list('kataInformal', flat=True)
        # arrSentimen = SentimenDB.objects.values_list('kataSentimen', flat=True)
        #
        # arrData = []
        # arrData.extend(arrKataFormal)
        # arrData.extend(arrFormalisasi)
        # arrData.extend(arrSentimen)
        # arrData = list(set(arrData))

        # arrData = []
        arrStopwords = StopwordsIDDB.objects.values_list('kataStopword', flat=True)
        # arrData.extend(arrStopwords)

        # dictKata = {}
        remove = string.punctuation
        remove = remove.replace("#", "")
        remove = remove.replace("@", "")

        name = request.FILES['dataset'].name
        typeFile = name.split('.')[1]
        if (typeFile == 'txt'):
            readers = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
        elif (typeFile == 'csv'):
            try:
                text = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
                readers = csv.reader(text)
            except:
                text = StringIO(request.FILES['dataset'].file.read().decode())
                readers = csv.reader(text)
        else:
            return render(request, 'JST/inputDataSentimen.html', {})

        dictKata = {} #berisi TF dari masing2 kata
        indexDoc = 0
        dictData = {} #berisi raw dokumen
        dfjKata = {} #berisi banyak dokumen yang memuat suatu kata
        arrCorpus =[] #array menyimpan file dari memori
        numDocs = 0

        #Memindahkan file dari memory ke array
        for reader in readers:
            kalimat = ''.join(reader)
            #kalimat = kalimat.translate(str.maketrans('', '', remove)).lower()
            arrCorpus.append(kalimat)
            numDocs += 1

        #Buat data untuk MI dan Formalisasi database
        arrDataMI = []
        formalisasi = FormalisasiKataDB.objects.values_list('kataInformal', 'kataFormal')
        kataFormalisasi = {}
        for i in range(0, len(formalisasi)):
            kataFormalisasi[str(formalisasi[i][0])] = str(formalisasi[i][1])

        #Menyimpan data mentahan dan formalisasi untuk ektraksi MI
        for reader in arrCorpus:
            #reader = ''.join(reader)
            line = str(reader).lower()
            baris = line.split()

            if (len(baris) > 0):
                dictData[indexDoc] = line
                indexDoc += 1

            if (len(baris) > 0):
                kalimat = ""
                for x in range(0, len(baris)):
                    if baris[x] in kataFormalisasi.keys():
                        baris[x] = kataFormalisasi[baris[x]]
                    kalimat = kalimat + " " + baris[x]
                arrDataMI.append(kalimat)

        #Hitung TF dari masing2 kata
        for line in arrDataMI:
            line = line.translate(str.maketrans('', '', remove)).lower()
            baris = line.split()
            if (len(baris) > 0):
                #TF untuk unigram
                for kata in baris:
                    if kata not in dictKata.keys():
                        dictKata[kata] = 1
                        dfjKata[kata] = 0
                    else:
                        dictKata[kata] += 1
                #TF untuk bigram
                for i in range(0, (len(baris) - 1)):
                    kata = baris[i] + " " + baris[i + 1]
                    if kata not in dictKata.keys():
                        dictKata[kata] = 1
                        dfjKata[kata] = 0
                    else:
                        dictKata[kata] += 1

        for line in arrDataMI:
            line = line.translate(str.maketrans('', '', remove)).lower()
            baris = line.split()
            if (len(baris) > 0):
                for i in range(0, len(baris) - 1):
                    kata = str(''.join(baris[i] + " " + baris[i + 1]))
                    baris.append(kata)
            # Menghitung dfj
            for kata in dictKata.keys():
                if kata in baris:
                    if (dfjKata[kata] == 0):
                        dfjKata[kata] = 1
                    else:
                        dfjKata[kata] += 1

        # Inisialisasi dan hitung tf-idf
        tfidfKata = {}

        # Cek stopwords
        stopwords = request.POST['stopwords']
        if (stopwords == 'yes'):
            for kata in arrStopwords:
                if kata in dictKata.keys():
                    # logging.warning(str(kata))
                    del dictKata[kata]

        for kata in dictKata.keys():
            # logging.warning(kata)
            if (dfjKata[kata] == numDocs):
                n = 0
            else:
                n = 1
            tfidfKata[kata] = dictKata[kata] * np.log(numDocs / (dfjKata[kata] + n))
            # logging.warning(str(kata) +" : "+str(tfidfKata[kata]))

        # arrKata = sorted(dictKata, key=dictKata.__getitem__, reverse=True)
        arrKata = sorted(tfidfKata, key=tfidfKata.__getitem__, reverse=True)

        # file.close()
        #arrKata = sorted(dictKata, key=dictKata.__getitem__, reverse=True)

        w = 0
        kata = {}
        for word in arrKata:
            kata[w] = word
            w += 1
        vocabSize = int(request.POST['vocabSize'])
        if (w > vocabSize):
            w = vocabSize

        #logging.warning(str(kata[0]))
        kalimat = str(kata[0])
        for i in range(1, w):
            kalimat = kalimat +","+kata[i]

        statusMI = request.POST['statusMI']
        if(statusMI == 'yes'):
            statusMI = True
        else:
            statusMI = False

        return render(request, 'JST/previewMI.html', {'dictData': dictData, 'kata': kata, 'jarak': range(0, w),
                                                     'kalimat': kalimat, 'lenCorpus' : indexDoc, 'name': name,
                                                     'statusMI': statusMI})
    else:
        return render(request, 'JST/inputDataMI.html', {})

def prosesMI(request):
    if request.method == 'POST':
        formalisasi = FormalisasiKataDB.objects.values_list('kataInformal', 'kataFormal')
        kataFormalisasi = {}
        for i in range(0, len(formalisasi)):
            kataFormalisasi[str(formalisasi[i][0])] = str(formalisasi[i][1])

        remove = string.punctuation
        remove = remove.replace("#", "")
        remove = remove.replace("@", "")
        remove = remove.replace(",", "")
        # print(remove)

        #Membuat lisr memuat kata2 MI Positive
        positiveMI = request.POST['positiveMI'].translate(str.maketrans('', '', remove)).lower()
        positiveMIFormalisasi = []
        positiveMIArr = positiveMI.split(',')
        for kata in positiveMIArr:
            katas = kata.split()
            kataBaru = ""
            for i in range(0, len(katas)):
                if katas[i] in kataFormalisasi.keys():
                    katas[i] = kataFormalisasi[katas[i]]
                    if (i == 0):
                        kataBaru = str(katas[i])
                    else:
                        kataBaru = str(kataBaru) +" "+str(katas[i])
                else:
                    if (i == 0):
                        kataBaru = str(katas[0])
                    else:
                        kataBaru = str(kataBaru) +" "+str(katas[i])
            positiveMIFormalisasi.append(kataBaru)


        #Membuat list yang memuat kata2 MI Negative
        negativeMI = request.POST['negativeMI'].translate(str.maketrans('', '', remove)).lower()
        negativeMIFormalisasi = []
        negativeMIArr = negativeMI.split(',')
        for kata in negativeMIArr:
            katas = kata.split()
            kataBaru = ""
            for i in range(0, len(katas)):
                if katas[i] in kataFormalisasi.keys():
                    katas[i] = kataFormalisasi[katas[i]]
                    if (i == 0):
                        kataBaru = str(katas[i])
                    else:
                        kataBaru = str(kataBaru) +" "+str(katas[i])
                else:
                    if (i == 0):
                        kataBaru = str(katas[0])
                    else:
                        kataBaru = str(kataBaru) +" "+str(katas[i])
            negativeMIFormalisasi.append(kataBaru)

        #Membuat list yang memuat dokumen2 dari corpus
        arrData = []
        arrDataRaw = []
        lenCorpus = int(request.POST['lenCorpus'])
        logging.warning(str(lenCorpus))
        #numRaw = 0
        #numProcess = 0
        for i in range(0, lenCorpus):
            kalimat = "kalimat_"+str(i)
            kalimat = request.POST[kalimat]
            kata = kalimat.translate(str.maketrans('', '', remove)).lower()
            baris = kata.split()
            if (len(baris) > 0):
                arrDataRaw.append(kalimat)
                #numRaw += 1
                # proses Formalisasi
                kalimatBaru = ""
                for x in range(0, len(baris)):
                    if baris[x] in kataFormalisasi.keys():
                        baris[x] = kataFormalisasi[baris[x]]
                    kalimatBaru = kalimatBaru + " " + baris[x]
                arrData.append(kalimatBaru)
                #numProcess += 1
        #logging.warning("Jumlah data awal : "+str(numRaw))
        #logging.warning("Jumlah data hasil proses : "+str(numProcess))
        # Memberikan nilai untuk hyperparameter dari User atau otomatis
        if (request.POST['alpha'] == ""):
            alpha = -1
        else:
            alpha = float(request.POST['alpha'])

        if (request.POST['beta'] == ""):
            beta = -1
        else:
            beta = float(request.POST['beta'])

        if (request.POST['gamma'] == ""):
            gamma = -1
        else:
            gamma = float(request.POST['gamma'])

        if(request.POST['topics'] == ""):
            topics = 1
        else:
            topics = int(request.POST['topics'])

        if(request.POST['iterasi'] == ""):
            iterasi = 1000
        else:
            iterasi = int(request.POST['iterasi'])

        # Cek stopwords
        stopwords = request.POST['stopwords']
        if (stopwords == 'yes'):
            statusStopwords = True
        else:
            statusStopwords = False

        statusLexicon = request.POST['FSL']
        if (statusLexicon == 'none'):
            statusFSL = False
            if (request.POST['filtered'] == ""):
                filtered = 0
            else:
                filtered = int(request.POST['filtered'])
        elif (statusLexicon == 'full'):
            statusFSL = True
            filtered = 0
        else:
            statusFSL = True
            if (request.POST['filtered'] == ""):
                filtered = 0
            else:
                filtered = int(request.POST['filtered'])

        # Mencari status dari file label untuk pengujian prior
        cekLabel = request.FILES.get('label', False)
        if (cekLabel != False):
            typeFile = (request.FILES['label'].name).split('.')[1]
            if (typeFile == 'txt'):
                labels = TextIOWrapper(request.FILES['label'].file, encoding='utf-8 ', errors='replace')
            elif (typeFile == 'csv'):
                try:
                    text = TextIOWrapper(request.FILES['label'].file, encoding='utf-8 ', errors='replace')
                    labels = csv.reader(text)
                except:
                    text = StringIO(request.FILES['dataset'].file.read().decode())
                    labels = csv.reader(text)
            else:
                return render(request, 'JST/inputDataMI.html', {})

            dictLabel = {}
            for key, label in enumerate(labels):
                label = int(''.join(label))
                dictLabel[key] = int(label)

            if (len(dictLabel) != len(arrData)):
                return render(request, 'JST/inputDataMI.html', {})

        #Lakukan Proses utuk MI
        simulationIteration = int(request.POST['iterasiSimulasi'])

        positiveTopics = []
        negativeTopics = []

        pi_dli = np.zeros((len(arrData), 2, simulationIteration))

        vocabSize = 0
        corpusSize = 0 #banyak kata dalam suatu corpus
        corpusLength = 0 #banyak dokumen dalam suatu corpus
        priorLabeled = 0
        aveDocSize = 0.0

        kataPositive = []
        kataNegative = []
        kalimatHasil = []

        waktuSimulasi = {}
        hyperparametersSimulasi = {}
        hyperparameter = ""
        dataSimulasi = []

        # Mengekstrak topic words untuk tiap label sentimen
        pengaliPeluang = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2500, 2000, 1500, 1200,
                          1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
        kali = 0

        for i in range(0, simulationIteration):
            positiveSimulasi = {}
            negativeSimulasi = {}
            name = "JST MI Simulation : %s " % str(simulationIteration)
            jst = modelJST(alpha, beta, gamma, topics, name, statusStopwords, statusFSL, filtered, iterasi,
                           positiveMIFormalisasi, negativeMIFormalisasi)
            jst.execute_model(arrData)
            waktuSimulasi[i] = jst.processTime
            hyperparametersSimulasi[i] = str(round(jst.alpha, 4)) + " * " + \
                                         str(round(jst.beta, 4)) + " * " + \
                                         str(round(jst.gamma, 4))
            for d in range(0, jst.numDocs):
                for l in range(0, jst.rangeSentiLabs):
                    pi_dli[d][l][i] = jst.pi_dl[d][l]

            if(i == 0):
                kalimatHasil = jst.arrData
                hyperparameter = str(jst.alpha) + ", " + str(jst.beta) + "," + str(jst.gamma)
                vocabSize = jst.vocabSize
                corpusSize = jst.corpusSize
                priorLabeled = jst.labelPrior
                aveDocSize = jst.aveDocLength
                kataPositive = jst.pdataset.labeledPositiveWords
                kataNegative = jst.pdataset.labeledNegativeWords
                corpusLength = jst.numDocs


            # JSON untuk topik positive
            for z in range(0, topics):

                words_probs = {}
                for w in range(0, vocabSize):
                    words_probs[w] = [w, jst.phi_lzw[1][z][w]]
                topicsWords = sorted(words_probs.items(), key=lambda item: item[1][1], reverse=True)

                for pengali in pengaliPeluang:
                    if (topicsWords[0][1][1] * pengali < 90):
                        kali = pengali
                        break

                positiveTopic = []
                for i in range(0, 40):
                    positiveTopic.append([jst.id2word[topicsWords[i][1][0]], int(round(topicsWords[i][1][1] * kali))])
                positiveSimulasi[z] = positiveTopic
            positiveTopics.append(positiveSimulasi)

            # JSON untuk topik negative
            for z in range(0, topics):
                words_probs = {}
                for w in range(0, vocabSize):
                    words_probs[w] = [w, jst.phi_lzw[0][z][w]]
                topicsWords = sorted(words_probs.items(), key=lambda item: item[1][1], reverse=True)

                for pengali in pengaliPeluang:
                    if (topicsWords[0][1][1] * pengali < 90):
                        kali = pengali
                        break

                negativeTopic = []
                for i in range(0, 40):
                    negativeTopic.append([jst.id2word[topicsWords[i][1][0]], int(round(topicsWords[i][1][1] * kali))])
                negativeSimulasi[z] = negativeTopic
            negativeTopics.append(negativeSimulasi)


        #Membuat JSON unutk hasil peluang sentimen tiap dokumen dalam tiap simulasi
        for d in range(0, corpusLength):
            data = {}
            data['kalimat'] = kalimatHasil[d]
            for i in range(0, simulationIteration):
                data['positive_'+str(i)] = pi_dli[d][1][i]
                data['negative_'+str(i)] = pi_dli[d][0][i]
                if(pi_dli[d][1][i] > pi_dli[d][0][i]):
                    label = 1
                elif(pi_dli[d][1][i] < pi_dli[d][0][i]):
                    label = 2
                else:
                    label = 0
                data['hasil_'+str(i)] = label
            dataSimulasi.append(data)
        jsonSimulasi = json.dumps(dataSimulasi)

        #Membuat ringkasan label sentimen
        sentimenSimulasi = []
        for i in range(0, simulationIteration):
            sentimenLabel = []
            for d in range(0, corpusLength):
                if (pi_dli[d][1][i] > pi_dli[d][0][i]):
                    label = 'positive'
                elif (pi_dli[d][1][i] < pi_dli[d][0][i]):
                    label = 'negative'
                else:
                    label = 'netral'
                sentimenLabel.append(label)
                sentimenLabel.append('total')
            sentimenTest = Counter(sentimenLabel)
            sentimenSimulasi.append(sentimenTest)
        jsonSentimen = json.dumps(sentimenSimulasi)
        jsonPositive = json.dumps(positiveTopics)
        jsonNegative = json.dumps(negativeTopics)


        if(cekLabel == False):
            name = "JST MI tanpa label Simulation : %s " % str(simulationIteration)
            #Membuat json untuk review simulasi
            arrReviewSimulasi = []
            for i in range(0, simulationIteration):
                arrRStemp = {}
                arrRStemp['waktu'] = waktuSimulasi[i]
                arrRStemp['hyperparameter'] = hyperparametersSimulasi[i]
                arrRStemp['positive'] = round(
                    (sentimenSimulasi[i]['positive']/sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['negative'] = round(
                    (sentimenSimulasi[i]['negative'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['objektif'] = round(
                    (sentimenSimulasi[i]['netral'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrReviewSimulasi.append(arrRStemp)
            jsonReviewSimulasi = json.dumps(arrReviewSimulasi)

            return render(request, 'JST/HasilJSTSimulasi.html', {'corpusLength': corpusLength,
                                                                 'name': name,
                                                                 'stopwordsStatus': statusStopwords,
                                                                 'lexiconStatus': statusLexicon + " (" + str(
                                                                     filtered) + ")",
                                                                 'hyperparameters': hyperparameter,
                                                                 'vocabSize': vocabSize,
                                                                 'corpusSize': corpusSize,
                                                                 'aveDocSize': aveDocSize,
                                                                 'priorLabeled': priorLabeled,
                                                                 'topics': topics,
                                                                 'iterasiGibbs': iterasi,
                                                                 'kataPositive': kataPositive,
                                                                 'kataNegative': kataNegative,
                                                                 'jsonSimulasi': jsonSimulasi,
                                                                 'iterasiSimulasi': simulationIteration,
                                                                 'jsonPositive': jsonPositive,
                                                                 'jsonNegative': jsonNegative,
                                                                 'jsonSentimen': jsonSentimen,
                                                                 'jsonReviewSimulasi': jsonReviewSimulasi,
                                                                 })
        else:
            name = "JST MI dengan label Simulation : %s " % str(simulationIteration)
            #Membuat pengukuran terhadap akurasi
            akurasiSimulasi = {}
            for i in range(0, simulationIteration):
                sumDocLabel = 0
                sumDocAkurasi = 0
                for d in range(0, len(arrData)):
                    if (pi_dli[d][1][i] > pi_dli[d][0][i]):
                        sentiLab = 1
                        sumDocLabel += 1
                    elif (pi_dli[d][1][i] < pi_dli[d][0][i]):
                        sentiLab = 2
                        sumDocLabel += 1
                    else:
                        sentiLab = -1

                    if (str(sentiLab) == str(dictLabel[i])):
                        sumDocAkurasi += 1
                akurasiSimulasi[i] = round((sumDocAkurasi / sumDocLabel) * 100, 2)

            #membuat json untuk review simulasi dengan nilai akurasi labelnya
            arrReviewSimulasi = []
            for i in range(0, simulationIteration):
                arrRStemp = {}
                arrRStemp['waktu'] = waktuSimulasi[i]
                arrRStemp['hyperparameter'] = hyperparametersSimulasi[i]
                arrRStemp['positive'] = round(
                    (sentimenSimulasi[i]['positive'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['negative'] = round(
                    (sentimenSimulasi[i]['negative'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['objektif'] = round(
                    (sentimenSimulasi[i]['netral'] / sentimenSimulasi[i]['total']) * 100, 2)
                arrRStemp['akurasi'] = akurasiSimulasi[i]
                arrReviewSimulasi.append(arrRStemp)
            jsonReviewSimulasi = json.dumps(arrReviewSimulasi)
            return render(request, 'JST/HasilJSTSimulasi.html', {'corpusLength': corpusLength,
                                                                 'name': name,
                                                                 'stopwordsStatus': statusStopwords,
                                                                 'lexiconStatus': statusLexicon + " (" + str(
                                                                     filtered) + ")",
                                                                 'dictLabel': dictLabel,
                                                                 'hyperparameters': hyperparameter,
                                                                 'vocabSize': vocabSize,
                                                                 'corpusSize': corpusSize,
                                                                 'aveDocSize': aveDocSize,
                                                                 'priorLabeled': priorLabeled,
                                                                 'iterasiSimulasi': simulationIteration,
                                                                 'topics': topics,
                                                                 'iterasiGibbs': iterasi,
                                                                 'kataPositive': kataPositive,
                                                                 'kataNegative': kataNegative,
                                                                 'jsonSimulasi': jsonSimulasi,
                                                                 'jsonPositive': jsonPositive,
                                                                 'jsonNegative': jsonNegative,
                                                                 'jsonSentimen': jsonSentimen,
                                                                 'jsonReviewSimulasi': jsonReviewSimulasi,
                                                                 })

def inputDataPelabelan(request):
    if request.method == 'POST':
        dictKalimat = {}
        name = request.FILES['dataset'].name
        typeFile = (request.FILES['dataset'].name).split('.')[1]
        if (typeFile == 'txt'):
            readers = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
        elif (typeFile == 'csv'):
            try:
                text = TextIOWrapper(request.FILES['dataset'].file, encoding='utf-8 ', errors='replace')
                readers = csv.reader(text)
            except:
                text = StringIO(request.FILES['dataset'].file.read().decode())
                readers = csv.reader(text)
        else:
            return render(request, 'JST/inputDataPelabelan.html', {})

        for key, reader in enumerate(readers):
            reader = ''.join(reader)
            dictKalimat[key] = reader
        corpusLength = len(dictKalimat)
        return render(request, 'JST/previewPelabelan.html', {'dictKalimat': dictKalimat,
                                                             'range': range(0, corpusLength),
                                                             'corpusLength': corpusLength,
                                                             'name': name})
    else:
        return render(request, 'JST/inputDataPelabelan.html', {})

def simpanPelabelan(request):
    sizeArrData = request.POST['corpusLength']
    sizeArrData = int(sizeArrData)

    unduhFile = request.POST['unduhFile']
    unduhFile = str(unduhFile)
    dataKalimat = {}
    dataLabel = {}

    if (unduhFile == 'dataset'):
        for i in range(0, sizeArrData):
            status = 'status_' + str(i)
            status = request.POST[status]
            if (status == 'spam'):
                pass
            else:
                kalimat = 'kalimat_' + str(i)
                kalimat = request.POST[kalimat]
                dataKalimat[i] = kalimat

        Kalimat = StringIO()
        for key in dataKalimat.keys():
            Kalimat.write(dataKalimat[key] + os.linesep)
        # Kalimat.write(str(countData))

        Kalimat.flush()
        Kalimat.seek(0)
        response = HttpResponse(FileWrapper(Kalimat), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=kalimat.csv'
        return response
    elif(unduhFile == 'label'):
        for i in range(0, sizeArrData):
            status = 'status_' + str(i)
            status = request.POST[status]
            if (status == 'spam'):
                pass
            elif (status == 'positive'):
                dataLabel[i] = 1
            elif (status == 'negative'):
                dataLabel[i] = 0
            elif (status == 'netral'):
                dataLabel[i] = -1

        Label = StringIO()
        for key in dataLabel.keys():
            Label.write(str(dataLabel[key]) + os.linesep)
        # Label.write(str(countData))

        Label.flush()
        Label.seek(0)
        response = HttpResponse(FileWrapper(Label), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=Label.txt'
        return response
    elif(unduhFile == 'full'):
        #spam = []
        # buffer1 = StringIO()
        # kalimat = csv.writer(buffer1, quoting=csv.QUOTE_NONE)

        # buffer2 = StringIO()
        # label = csv.writer(buffer2, quoting=csv.QUOTE_NONE)

        # buffer3 = StringIO()
        # spam = csv.writer(buffer3, quoting=csv.QUOTE_NONE)

        # arrKalimat = []
        # arrLabel =[]
        # arrSpam = []

        spam = StringIO()
        label = StringIO()
        kalimat = StringIO()

        for i in range(0, sizeArrData):
            status = 'status_' + str(i)
            status = request.POST[status]
            if (status == 'spam'):
                teks = request.POST['kalimat_'+str(i)]
                if(len(teks.split()) > 0):
                    spam.write(teks + os.linesep)

                # arrSpam.append(str(teks))
            elif(status == 'positive'):
                senLabel = 1
                teks = request.POST['kalimat_'+str(i)]
                label.write(str(senLabel) + os.linesep)
                kalimat.write(teks + os.linesep)

                # arrLabel.append(str(senLabel))
                # arrKalimat.append(str(teks))
            elif(status == 'negative'):
                senLabel = 2
                teks = request.POST['kalimat_' + str(i)]
                label.write(str(senLabel) + os.linesep)
                kalimat.write(teks + os.linesep)

                # arrLabel.append(str(senLabel))
                # arrKalimat.append(str(teks))
            elif(status == 'netral'):
                senLabel = -1
                teks = request.POST['kalimat_' + str(i)]
                label.write(str(senLabel) + os.linesep)
                kalimat.write(teks + os.linesep)

                # arrLabel.append(str(senLabel))
                # arrKalimat.append(str(teks))

        # label.writerows(arrLabel)
        # kalimat.writerows(arrKalimat)
        # spam.writerows(arrSpam)

        outfile = BytesIO()
        zip = ZipFile(outfile, 'w')

        # buffer1.flush()
        # buffer2.flush()
        # buffer3.flush()
        #
        # buffer1.seek(0)
        # buffer2.seek(0)
        # buffer3.seek(0)

        spam.flush()
        spam.seek(0)

        label.flush()
        label.seek(0)

        kalimat.flush()
        kalimat.seek(0)

        zip.writestr("label.csv", label.getvalue())
        zip.writestr("kalimat.csv", kalimat.getvalue())
        zip.writestr("spam.csv", spam.getvalue())

        #fix for linux zip files
        for file in zip.filelist:
            file.create_system = 0

        zip.close()

        response = HttpResponse(outfile.getvalue(), content_type='application/octet-stream')
        response['Content-Disposition'] = 'attachment; filename=hasil.zip'

        return response

def cekKoneksi():
    try:
        url = 'http://kateglo.com'
        requests.get(url)
        logging.warning("Koneksi sukses")
        return True
    except requests.ConnectionError:
        return False


