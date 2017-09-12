from django.shortcuts import render, redirect
from .models import SentimenDB, FormalisasiKataDB, KataFormalDB, StopwordsIDDB
import string, time, random, os, logging, csv, json, requests
from zipfile import ZipFile
import numpy as np
import scipy.special as scp
from io import StringIO, TextIOWrapper, BytesIO
from builtins import str
from django.http import HttpResponse
from wsgiref.util import FileWrapper
from collections import Counter
from preprocess.formalisasi import correction
from datetime import datetime
class document(object):
    def __init__(self):
        self.length = 0
        self.words = {}
        self.priorSentiLabels = {}
        self.docID = ""

class dataset(object):
    def __init__(self):
        self.word2atr = {}
        self.sentiLex = {}
        self.freqWord = {}
        self.id2word = {}
        self.pdocs = {}

        self.numDocs = 0
        self.aveDocLength = 0.0000  # average document length
        self.vocabSize = 0
        self.corpusSize = 0
        self.numVocabs = 0
        self.maxLength = 0

        self.labeledPositiveWords = []
        self.labeledNegativeWords = []

        self.arrData = []

    def sentiFile(self, statusFSL, positiveMI=None, negativeMI=None):
        if(statusFSL == True):
            sentimen = SentimenDB.objects.values_list('kataSentimen', 'sentiLab', 'priorNetral', 'priorPositive', 'priorNegative')
            for senti in sentimen:
                #Recoding label sentimen dari DB model dimana 1:positive, 2:negative -> 1:positive, 0:negatitve
                if(str(senti[1]) == '2'):
                    label = 0
                elif(str(senti[1]) == '1'):
                    label = 1
                self.sentiLex[senti[0]] = [label, [float(senti[2]), float(senti[3]), float(senti[4])]]
        print("Banyak Subjectivity Lexicon : "+str(len(self.sentiLex)))
        if(positiveMI != None):
            for kata in positiveMI:
                print("SentiPOS : "+str(kata))
                self.sentiLex[str(kata)] = [1, [0.05, 0.90, 0.05]]
        if(negativeMI != None):
            for kata in negativeMI:
                print("SentiNEG : "+str(kata))
                self.sentiLex[str(kata)] = [0, [0.05, 0.05, 0.05]]
        #sentimenDB = self.sentiLex
        return self.sentiLex

    def tokenisasi(self, teks):
        dataAwal = self.sentiLex.keys()
        arrNegasi = ['tidak', 'bukan', 'jangan', 'tak']
        kalimat = teks.lower()
        arrUji = kalimat.split()
        setBigram = False
        setBigramNegasi = False
        arrHasil = []
        for i in range(0, len(arrUji)):
            if(setBigram == True):
                setBigram = False
                pass
            elif(setBigramNegasi == True):
                setBigramNegasi = False
                pass
            else:
                if(i < (len(arrUji) - 1)):
                    kataAwal = arrUji[i]
                    kataAkhir = arrUji[i+1]
                    kataGabungan = kataAwal + " " + kataAkhir
                    if kataAwal in arrNegasi:
                        if(i < (len(arrUji) - 2)):
                            cekKata = arrUji[i+1] +" "+ arrUji[i+2]
                            if(cekKata in dataAwal):
                                token = kataAwal + " " + cekKata
                                arrHasil.append(token)
                                setBigram = True
                                setBigramNegasi = True
                        else:
                            token = kataGabungan
                            arrHasil.append(token)
                            setBigram = True
                    elif kataGabungan in dataAwal:
                        token = kataGabungan
                        arrHasil.append(token)
                        setBigram = True
                    elif kataAwal in dataAwal:
                        token = kataAwal
                        arrHasil.append(token)
                    else:
                        token = kataAwal
                        arrHasil.append(token)
                else:
                    token = arrUji[i]
                    arrHasil.append(token)
        # print(arrHasil)
        return arrHasil
        
    def readDataStream(self, arrData, statusStopwords, filtered):
        labelPOS = 0
        labelNEG = 0
        word = 0
        # Inisialisasi kata negasi
        daftarNegasi = ['tidak', 'bukan', 'tak', 'jangan']
        # idWord = self.pdataset.pdocs[d].words[t]
        # teks = self.id2word[idWord]
        # kata = teks.split()
        # if(len(kata) == 3):
        #     cekKata = 
        # elif (len(kata) == 2):
        #     pass
        # else:
        #     pass
        stopwords = StopwordsIDDB.objects.values_list('kataStopword', flat=True)
        stopwords = list(stopwords)
        filteredLimit = filtered

        # arrKataSebelum = []
        # arrKataTokenisasi = []

        #Menghapus sentiment lexicon yang ada di stopwords
        for kata in self.sentiLex.keys():
            try:
                stopwords.remove(kata)
            except ValueError:
                continue
                

        #Untuk menghitung frekuensi kemunculan kata untuk Model Prior Filtered Subjectivity Lexicon
        # self.freqWord['lorem'] = 0 #untuk mencegah genap
        for baris in arrData:
            # arrKataSebelum.append(len(baris.split()))
            
            ### TOKENISASI ###
            # line = self.tokenisasi(baris)

            line = baris.split()
            barisLength = len(line)
            for i in range(0, barisLength):
                if line[i] not in self.freqWord.keys():
                    self.freqWord[str(line[i])] = 1
                else:
                    self.freqWord[str(line[i])] += 1
        #Proses membaca corpus dengan keterangan lexicon
        idx = 0
        
        for baris in arrData:
            #logging.warning(str(baris))
            self.pdoc = document()

            ### TOKENISASI ###
            # line = self.tokenisasi(baris)

            line = baris.split()
            #print(line)
            # arrKataTokenisasi.append(len(line))
            

            #Checking stopwords
            if(statusStopwords == True):
                lineTemp = []
                for stopword in stopwords:
                    while True:
                        try:
                            line.remove(stopword)
                            lineTemp.append(stopword)
                        except ValueError:
                            break
                if(len(line) == 0):
                    line = lineTemp

            
            # if(len(line) % 2 == 0):
            #     line.append('lorem')
            docLength = len(line)

            if (docLength > self.maxLength):
                self.maxLength = docLength

            if (docLength > 0):
                self.arrData.append(baris)
                self.corpusSize += docLength
                #self.pdoc.length = docLength
                self.pdoc.docID = ("doc" + str(self.numDocs))
                self.pdoc.length = docLength
                self.numDocs += 1
                # Generate ID for tokens in the corpus, assign with voabulary id
                for k in range(0, docLength):
                    priorSenti = -1
                    if (line[k] not in self.word2atr.keys()):
                        word += 1
                        if(self.freqWord[str(line[k])] > filteredLimit):
                            if (line[k] in self.sentiLex.keys()):
                                #print(str(line[k])+" - "+str(self.sentiLex[str(line[k])][0]))
                                self.word2atr[str(line[k])] = [self.numVocabs, self.sentiLex[str(line[k])][0],
                                                               self.sentiLex[str(line[k])][1]]
                                self.pdoc.words[k] = self.numVocabs                                     
                                self.pdoc.priorSentiLabels[k] = self.word2atr[str(line[k])][1]
                                if(self.word2atr[str(line[k])][1] == 1):
                                    labelPOS += 1
                                elif (self.word2atr[str(line[k])][1] == 0):
                                    labelNEG += 1
                                #print(str(line[k]) + " - " +str(self.word2atr[str(line[k])][1]))

                                if(self.word2atr[str(line[k])][1] == 1):
                                    self.labeledPositiveWords.append(str(line[k]))
                                elif(self.word2atr[str(line[k])][1] == 0):
                                    self.labeledNegativeWords.append(str(line[k]))

                                self.id2word[self.numVocabs] = str(line[k])


                                self.numVocabs += 1
                            else:
                                # Memberikan label sentimen untuk kata negasi
                                arrKata = line[k].split()
                                if arrKata[0] in daftarNegasi:
                                    kataAkhir = ""
                                    if(len(arrKata) == 2):
                                        kataAkhir = arrKata[1]
                                    elif(len(arrKata) == 3):
                                        kataAkhir = arrKata[1] +" "+arrKata[2]

                                    if (kataAkhir in self.sentiLex.keys()):
                                        # print("Uji coba : "+kataAkhir)
                                        label = self.sentiLex[str(kataAkhir)][0]
                                        # print(str(label))
                                        if(label == 1):
                                            priorSenti = 0
                                        elif(label == 0):
                                            priorSenti = 1
                                #print(str(line[k])+" - "+ str(priorSenti))
                                #Akhir kasus untuk kata negasi
                                self.word2atr[str(line[k])] = [self.numVocabs, priorSenti, [1, 1, 1]]
                                self.pdoc.words[k] = self.numVocabs
                                self.pdoc.priorSentiLabels[k] = priorSenti

                                self.id2word[self.numVocabs] = str(line[k])

                                self.numVocabs += 1
                        else:
                            self.word2atr[str(line[k])] = [self.numVocabs, priorSenti, [1, 1, 1]]
                            self.pdoc.words[k] = self.numVocabs
                            self.pdoc.priorSentiLabels[k] = priorSenti

                            self.id2word[self.numVocabs] = str(line[k])

                            self.numVocabs += 1
                    else:
                        self.pdoc.words[k] = self.word2atr[str(line[k])][0]
                        self.pdoc.priorSentiLabels[k] = self.word2atr[str(line[k])][1]

            self.pdocs[idx] = self.pdoc
            idx += 1
        
        print("Banyak prior Positive : "+str(labelPOS))
        print("Banyak prior Negative : "+str(labelNEG))
        print("Banyak kata : "+str(word))
        print("Banyak Dokumen : "+str(self.numDocs))

        self.vocabSize = len(self.word2atr)
        self.aveDocLength = self.corpusSize / self.numDocs
        # for i in range(0, len(arrKataSebelum)):
        #     print(str(i)+" adalah "+str(arrKataSebelum[i])+" - "+str(arrKataTokenisasi[i]))

class modelJST(object):
    def __init__(self, alpha, beta, gamma, topics, name, statusStopwords, statusFSL, filtered, iterasi, positiveMI=None, negativeMI=None):
        if(positiveMI == None and negativeMI == None):
            self.word2atr = {}
            self.sentiLex = {}
            self.id2word = {}
            self.arrData = []

            self.numTopics = topics
            self.rangeSentiLabs = 2
            self.vocabSize = 0
            self.numDocs = 0
            self.corpusSize = 0
            self.aveDocLength = 0

            self.niters = iterasi  # 1000
            self.liter = 0
            self.savestep = 200  # 200
            self.twords = 20
            self.updateParaStep = 40  # 40

            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.name = name
            self.statusStopwords = statusStopwords
            self.statusFSL = statusFSL
            self.filtered = filtered

            self.positiveMI = None
            self.negativeMI = None
        elif(positiveMI != None and negativeMI != None):
            self.word2atr = {}
            self.sentiLex = {}
            self.id2word = {}
            self.arrData = []

            self.numTopics = topics
            self.rangeSentiLabs = 2
            self.vocabSize = 0
            self.numDocs = 0
            self.corpusSize = 0
            self.aveDocLength = 0

            self.niters = iterasi  # 1000
            self.liter = 0
            self.savestep = 200  # 200
            self.twords = 20
            self.updateParaStep = 40  # 40

            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.name = name
            self.statusStopwords = statusStopwords
            self.statusFSL = statusFSL
            self.filtered = filtered

            self.positiveMI = positiveMI
            self.negativeMI = negativeMI

    def execute_model(self, arrData):
        start = time.time()
        # definisi model dataset dari kelas dataset
        self.pdataset = dataset()
        # Mengeluarkan file berisi prior sentimen dari database
        self.sentiLex = self.pdataset.sentiFile(self.statusFSL, self.positiveMI, self.negativeMI)
        # Membuat dataset dengan masukkan array
        self.pdataset.readDataStream(arrData, self.statusStopwords, self.filtered)

        # Memanggil kamus kata dengan attribut dan kta dgn id
        self.word2atr = self.pdataset.word2atr
        self.id2word = self.pdataset.id2word
        self.arrData = self.pdataset.arrData

        #for id in self.id2word.keys():
        #   print("Kata : "+ str(self.id2word[int(id)]))

        # Proses pemanggilan awal
        self.initializing_parameter()

        # Proses estimasi awal
        self.initializing_estimasi()

        # Proses estimasi
        self.estimasi_model()

        end = time.time()
        self.processTime = end - start

        # if (arrLabel == None):
        #     start = time.time()
        #     # definisi model dataset dari kelas dataset
        #     self.pdataset = dataset()
        #     # Mengeluarkan file berisi prior sentimen dari database
        #     self.sentiLex = self.pdataset.sentiFile(self.positiveMI, self.negativeMI)
        #     # Membuat dataset dengan masukkan array
        #     self.pdataset.readDataStream(arrData, self.statusStopwords, self.filtered)
        #
        #     # Memanggil kamus kata dengan attribut dan kta dgn id
        #     self.word2atr = self.pdataset.word2atr
        #     self.id2word = self.pdataset.id2word
        #
        #     # Proses pemanggilan awal
        #     self.initializing_parameter()
        #
        #     # Proses estimasi awal
        #     self.initializing_estimasi()
        #
        #     # Proses estimasi
        #     self.estimasi_model()

        #     myfile = StringIO()
        #     myfile.write("Nilai alpha : " + str(self.alpha) + os.linesep)
        #     myfile.write("Nilai beta : " + str(self.beta) + os.linesep)
        #     myfile.write("Nilai gamma : " + str(self.gamma) + os.linesep)
        #     myfile.write("Document mean : " + str(self.aveDocLength) + os.linesep)
        #     myfile.write("Filtered Subjectivity Lexicon : "+str(self.filtered)+os.linesep)
        #     myfile.write("Stopwords : "+str(self.statusStopwords)+os.linesep)
        #     myfile.write("Iterasi : " + str(self.niters) + os.linesep)
        #     myfile.write("Update iterasi : " + str(self.savestep))
        #     myfile.write(os.linesep)
        #
        #     for d in range(0, self.numDocs):
        #         myfile.write("dokumen ke : " + str(d) + os.linesep)
        #         # myfile.write("Gammasum dokumen ke " + str(d) +" : " + str(self.gammaSum_d[d]))
        #         myfile.write(str(self.pdataset.pdocs[d].length) + os.linesep)
        #         myfile.write("Sentimen Netral : " + str(self.pi_dl[d][0]) + os.linesep)
        #         myfile.write("Sentimen Positive : " + str(self.pi_dl[d][1]) + os.linesep)
        #         myfile.write("Sentimen Negative : " + str(self.pi_dl[d][2]) + os.linesep)
        #         # for l in range(0, self.rangeSentiLabs):
        #         #    myfile.write("Nilai dari alphaSUm : "+ str(self.alphaSum_l[l])+ os.linesep)
        #         myfile.write(os.linesep)
        #     end3 = time.time()
        #     myfile.write(str(end3 - start))
        #     myfile.flush()
        #     myfile.seek(0)
        #
        #     response = HttpResponse(FileWrapper(myfile), content_type='text/csv')
        #     response['Content-Disposition'] = 'attachment; filename=JST.txt'
        #     return response
        # else:
        #     start = time.time()
        #     arrLabel = arrLabel
        #     akurasi = 0
        #     numDoc = 0
        #     # definisi model dataset dari kelas dataset
        #     self.pdataset = dataset()
        #     # Mengeluarkan file berisi prior sentimen dari database
        #     self.sentiLex = self.pdataset.sentiFile()
        #     # Membuat dataset dengan masukkan array
        #     self.pdataset.readDataStream(arrData, self.statusStopwords, self.filtered)
        #
        #     # Memanggil kamus kata dengan attribut dan kta dgn id
        #     self.word2atr = self.pdataset.word2atr
        #     self.id2word = self.pdataset.id2word
        #
        #     # Proses pemanggilan awal
        #     self.initializing_parameter()
        #
        #     # Proses estimasi awal
        #     self.initializing_estimasi()
        #
        #     # Proses estimasi
        #     self.estimasi_model()
        #
        #     myfile = StringIO()
        #
        #     myfile.write("Jenis : "+str(self.name)+os.linesep)
        #     myfile.write("Nilai alpha : " + str(self.alpha) + os.linesep)
        #     myfile.write("Nilai beta : " + str(self.beta) + os.linesep)
        #     myfile.write("Nilai gamma : " + str(self.gamma) + os.linesep)
        #     myfile.write("Filtered Subjectivity Lexicon : " + str(self.filtered) + os.linesep)
        #     myfile.write("Stopwords : " + str(self.statusStopwords) + os.linesep)
        #     myfile.write("Document mean : " + str(self.aveDocLength) + os.linesep)
        #     myfile.write("Banyak kata berlabel : " + str(self.labelPrior) + os.linesep)
        #     myfile.write("Banyak jenis kata (formalisasi) : " + str(len(self.word2atr)) + os.linesep)
        #     myfile.write("Banyak dokumen : " + str(self.numDocs) + os.linesep)
        #     myfile.write(os.linesep)
        #
        #     for z in range(0, self.numTopics):
        #         myfile.write("Alpha untuk topik ke - " + str(z) + " : " + str(self.alpha_temp[z]) + os.linesep)
        #     myfile.write("Alpha total : " + str(self.alphaSum_l[1]) + os.linesep)
        #     myfile.write(os.linesep)
        #
        #     outRange = 0
        #
        #     for d in range(0, self.numDocs):
        #         myfile.write("dokumen ke : " + str(d) + os.linesep)
        #         # myfile.write("Gammasum dokumen ke " + str(d) +" : " + str(self.gammaSum_d[d]))
        #         myfile.write(str(self.pdataset.pdocs[d].length) + os.linesep)
        #         myfile.write("Sentimen Netral : " + str(self.pi_dl[d][0]) + os.linesep)
        #         myfile.write("Sentimen Positive : " + str(self.pi_dl[d][1]) + os.linesep)
        #         myfile.write("Sentimen Negative : " + str(self.pi_dl[d][2]) + os.linesep)
        #
        #         if (self.pi_dl[d][1] > self.pi_dl[d][2] and self.pi_dl[d][1] > self.pi_dl[d][0]):
        #             label = 1
        #             numDoc += 1
        #         elif(self.pi_dl[d][1] > self.pi_dl[d][2]):
        #             label = 1
        #             numDoc += 1
        #             outRange += 1
        #         elif(self.pi_dl[d][2] > self.pi_dl[d][1] and self.pi_dl[d][2] > self.pi_dl[d][0]):
        #             label = 0
        #             numDoc += 1
        #         elif(self.pi_dl[d][2] > self.pi_dl[d][1]):
        #             label = 0
        #             numDoc += 1
        #             outRange += 1
        #         else:
        #             label = 0
        #             numDoc += 1
        #             outRange += 1
        #
        #         if (label == arrLabel[d]):
        #             akurasi += 1
        #         # for l in range(0, self.rangeSentiLabs):
        #         #    myfile.write("Nilai dari alphaSUm : "+ str(self.alphaSum_l[l])+ os.linesep)
        #         myfile.write(os.linesep)
        #     myfile.write("Akurasi terhadap label : " + str(akurasi / numDoc) + os.linesep)
        #     myfile.write("Lari dari acuan pelabelan : " + str(outRange) + os.linesep)
        #     end3 = time.time()
        #     myfile.write("Waktu proses : " + str(end3 - start))
        #     myfile.flush()
        #     myfile.seek(0)
        #
        #     response = HttpResponse(FileWrapper(myfile), content_type='text/csv')
        #     response['Content-Disposition'] = 'attachment; filename=JST.txt'
        #     return response

    def initializing_parameter(self):
        self.numDocs = self.pdataset.numDocs
        self.vocabSize = self.pdataset.vocabSize
        self.corpusSize = self.pdataset.corpusSize
        self.aveDocLength = self.pdataset.aveDocLength

        # Membentuk model masing - masing fungsi
        self.nd = np.zeros((self.numDocs))
        self.ndl = np.zeros((self.numDocs, self.rangeSentiLabs))
        self.ndlz = np.zeros((self.numDocs, self.rangeSentiLabs, self.numTopics))
        self.nlzw = np.zeros((self.rangeSentiLabs, self.numTopics, self.vocabSize))
        self.nlz = np.zeros((self.rangeSentiLabs, self.numTopics))

        # Posterior terhadap peluang dari masing2 dokumen
        self.p = np.zeros((self.rangeSentiLabs, self.numTopics))

        # Memodelkan paramater
        self.pi_dl = np.zeros((self.numDocs, self.rangeSentiLabs))
        self.theta_dlz = np.zeros((self.numDocs, self.rangeSentiLabs, self.numTopics))
        self.phi_lzw = np.zeros((self.rangeSentiLabs, self.numTopics, self.vocabSize))

        # Menginisiasikan nilai alpha
        if (self.alpha <= 0):
            self.alpha = (self.aveDocLength) / (self.rangeSentiLabs * self.numTopics)

        # Mengisikan nilai alpha ke model paramter
        self.alpha_lz = np.empty((self.rangeSentiLabs, self.numTopics))
        self.alpha_lz.fill(self.alpha)

        self.alphaSum_l = np.zeros((self.rangeSentiLabs))

        for l in range(0, self.rangeSentiLabs):
            for z in range(0, self.numTopics):
                self.alphaSum_l[l] += self.alpha_lz[l][z]

        # Menginisiasikan nilai betha
        if (self.beta <= 0.0):
            self.beta = 0.01

        self.beta_lzw = np.empty((self.rangeSentiLabs, self.numTopics, self.vocabSize))
        self.beta_lzw.fill(self.beta)

        self.betaSum_lz = np.zeros((self.rangeSentiLabs, self.numTopics))

        # Menginisisikan nilai gamma
        if (self.gamma <= 0):
            self.gamma = (self.aveDocLength) / self.rangeSentiLabs

        self.gamma_dl = np.empty((self.numDocs, self.rangeSentiLabs))
        self.gamma_dl.fill(self.gamma)

        self.gammaSum_d = np.zeros((self.numDocs))
        for d in range(0, self.numDocs):
            for l in range(0, self.rangeSentiLabs):
                self.gammaSum_d[d] += self.gamma_dl[d][l]

        # Mentransformasi kata2 terhadap label sentimen masing2
        self.lambda_lw = np.ones((self.rangeSentiLabs, self.vocabSize))

        for word in self.sentiLex.keys():
            for j in range(0, self.rangeSentiLabs):
                if (word in self.word2atr.keys()):
                    self.lambda_lw[j][self.word2atr[str(word)][0]] = self.sentiLex[str(word)][1][j]

        for l in range(0, self.rangeSentiLabs):
            for z in range(0, self.numTopics):
                for r in range(0, self.vocabSize):
                    self.beta_lzw[l][z][r] = self.beta_lzw[l][z][r] * self.lambda_lw[l][r]
                    self.betaSum_lz[l][z] += self.beta_lzw[l][z][r]
                #logging.warning("Nilai beta awal label ke "+str(l)+" topik ke "+str(z)+" : "+str(self.betaSum_lz[l][z]))

    def initializing_estimasi(self):
        # Menginisialisasikan topik ke setiap dokumen
        self.z = np.empty((self.numDocs, self.pdataset.maxLength))
        self.z.fill(0)

        # Menginisalisasikan label ke setiap dokumen
        self.l = np.empty((self.numDocs, self.pdataset.maxLength))
        self.l.fill(0)
        self.labelPrior = 0
        for d in range(0, self.numDocs):
            docLength = self.pdataset.pdocs[d].length
            for t in range(0, docLength):
                if (self.pdataset.pdocs[d].priorSentiLabels[t] > -1):
                    # Memasukkan label sentimen dari prior ke model
                    sentiLab = self.pdataset.pdocs[d].priorSentiLabels[t]
                    self.labelPrior += 1
                else:
                    # bila kata tidak memiliki prior dari database dan untuk bigram dan negasi bigram
                    # sentiLab = int(round(random.uniform(0, 1) * self.rangeSentiLabs))
                    # if (sentiLab == self.rangeSentiLabs):
                    #     sentiLab = round(sentiLab - 1)

                    sentiLab = np.random.randint(self.rangeSentiLabs)
                    # print("NIlai sentilab : "+str(sentiLab))

                # self.l[d][t] = int(round(sentiLab))
                self.l[d][t] = int(sentiLab)

                # Meninisialisasikan topik secara random
                # topic = int(round(random.uniform(0, 1) * self.numTopics))
                
                # if (topic == self.numTopics): topic = (topic - 1)
                # self.z[d][t] = int(round(topic))

                topic = np.random.randint(self.numTopics)
                self.z[d][t] = int(topic)

                # model count assignment
                self.nd[d] += 1
                self.ndl[d][sentiLab] += 1
                self.ndlz[d][sentiLab][topic] += 1
                self.nlzw[sentiLab][topic][self.pdataset.pdocs[d].words[t]] += 1
                self.nlz[sentiLab][topic] += 1

    def estimasi_model(self):
        self.countUpdateParameter = 0
        for self.liter in range(0, self.niters):
            #logging.warning("iterasi ke : "+str(self.liter))
            for m in range(0, self.numDocs):
                for n in range(0, self.pdataset.pdocs[m].length):
                    sentiLab = int(round(self.l[m][n]))
                    topic = int(round(self.z[m][n]))

                    # Mengoptimasi topik dan label dari kata
                    sentiLab, topic = self.sampling(m, n, sentiLab, topic)

                    self.l[m][n] = int(round(sentiLab))
                    self.z[m][n] = int(round(topic))
            if((self.liter % 10) == 0):
                logging.warning("Waktu : "+ str(datetime.now()))
                logging.warning(
                    "Nilai peluang untuk label ke " + str(0) +" iterasi ke " + str(self.liter) + " : " +str(
                        self.p[0][0]))
                logging.warning(
                    "Nilai peluang untuk label ke " + str(1) + " iterasi ke " + str(self.liter) + " : " + str(
                        self.p[1][0]))
                # logging.warning(
                #     "Nilai peluang untuk label ke " + str(2) + " iterasi ke " + str(self.liter) + " : " + str(
                #         self.p[2][0]))
                logging.warning(' ')

            # if (self.updateParaStep > 0 and self.liter % self.updateParaStep == 0):
            #     self.update_Parameters()

            if (self.savestep > 0 and self.liter % self.savestep == 0):
                if (self.liter == self.niters): break

                # print("Saving the model at iteratiot '%d' \n" % self.liter)
                self.compute_pi_dl()
                self.compute_theta_dlz()
                self.compute_phi_lzw()

        self.compute_pi_dl()
        self.compute_theta_dlz()
        self.compute_phi_lzw()

    def compute_pi_dl(self):
        for d in range(0, self.numDocs):
            for l in range(0, self.rangeSentiLabs):
                self.pi_dl[d][l] = (self.ndl[d][l] + self.gamma_dl[d][l]) / (self.nd[d] + self.gammaSum_d[d])

    def compute_theta_dlz(self):
        for d in range(0, self.numDocs):
            for l in range(0, self.rangeSentiLabs):
                for z in range(0, self.numTopics):
                    self.theta_dlz[d][l][z] = (self.ndlz[d][l][z] + self.alpha_lz[l][z]) / (self.ndl[d][l] + self.alphaSum_l[l])

    def compute_phi_lzw(self):
        for l in range(0, self.rangeSentiLabs):
            for z in range(0, self.numTopics):
                for r in range(0, self.vocabSize):
                    self.phi_lzw[l][z][r] = (self.nlzw[l][z][r] + self.beta_lzw[l][z][r]) / (self.nlz[l][z] + self.betaSum_lz[l][z])

    def sampling(self, m, n, sentiLab, topic):
        w = self.pdataset.pdocs[m].words[n]
        sentiLab = sentiLab
        # print("sentimen : "+str(sentiLab))
        topic = topic
        # print("topik : "+str(topic))

        self.nd[m] -= 1
        self.ndl[m][sentiLab] -= 1
        self.ndlz[m][sentiLab][topic] -= 1
        self.nlzw[sentiLab][topic][w] -= 1
        self.nlz[sentiLab][topic] -= 1

        # do multinomial sampling via cumulative method
        for l in range(0, self.rangeSentiLabs):
            for k in range(0, self.numTopics):
                self.p[l][k] = ((self.nlzw[l][k][w] + self.beta_lzw[l][k][w]) / (self.nlz[l][k] + self.betaSum_lz[l][k])) * \
                               ((self.ndlz[m][l][k] + self.alpha_lz[l][k]) / (self.ndl[m][l] + self.alphaSum_l[l])) * \
                               ((self.ndl[m][l] + self.gamma_dl[m][l]) / (self.nd[m] + self.gammaSum_d[m]))
                #logging.warning("Nilai peluang untuk label ke "+str(l)+" iterasi ke "+str(self.liter)+" : "+str(self.p[l][k]))

        # accumulate multinomial parameters
        for l in range(0, self.rangeSentiLabs):
            for z in range(0, self.numTopics):
                if (z == 0):
                    if (l == 0):
                        continue
                    else:
                        self.p[l][z] += self.p[l - 1][self.numTopics - 1]  # accumulate the sum of the previous array
                else:
                    self.p[l][z] += self.p[l][z - 1]

        # probability normalization
        u = random.uniform(0, 1) * self.p[self.rangeSentiLabs - 1][self.numTopics - 1]

        # sample sentiment label l, where l \in [0, S-1]
        loopBreak = False
        for sentiLab in range(0, self.rangeSentiLabs):
            for topic in range(0, self.numTopics):
                if (self.p[sentiLab][topic] > u):
                    loopBreak = True
                    break
            if (loopBreak == True):
                break

        if (sentiLab == self.rangeSentiLabs): sentiLab = int(round(self.rangeSentiLabs - 1))
        if (topic == self.numTopics): topic = int(round(self.numTopics - 1))

        # add estiamted 'z' and 'l' to count variable
        self.nd[m] += 1
        self.ndl[m][sentiLab] += 1
        self.ndlz[m][sentiLab][topic] += 1
        self.nlzw[sentiLab][topic][self.pdataset.pdocs[m].words[n]] += 1
        self.nlz[sentiLab][topic] += 1

        return sentiLab, topic

    def update_Parameters(self):
        self.data = np.zeros((self.numTopics, self.numDocs))
        self.alpha_temp = np.zeros((self.numTopics))
        # self.nanCondions = False
        # update alpha
        for l in range(0, self.rangeSentiLabs):
            for z in range(0, self.numTopics):
                for d in range(0, self.numDocs):
                    self.data[z][d] = self.ndlz[d][l][z]

            for z in range(0, self.numTopics):
                self.alpha_temp[z] = self.alpha_lz[l][z]

            self.polya_fit_simple(self.data, self.alpha_temp, self.numTopics, self.numDocs)

            # update alpha
            self.alphaSum_l[l] = 0.0
            for z in range(0, self.numTopics):
                self.alpha_lz[l][z] = self.alpha_temp[z]
                self.alphaSum_l[l] += self.alpha_lz[l][z]

    def polya_fit_simple(self, data, alpha, numTopics, numDocs):
        K = numTopics
        nSample = numDocs
        polya_iter = 100000
        sat_state = False
        # mp.dps = 8

        old_alpha = np.zeros((K))
        data_row_sum = np.zeros((nSample))

        for i in range(0, nSample):
            for k in range(0, K):
                # data_row_sum[i] +=  mp.mpf(data[k][i])
                data_row_sum[i] += data[k][i]

        for i in range(0, polya_iter):
            sum_alpha_old = 0.0

            for k in range(0, K):
                old_alpha[k] = alpha[k]
            for k in range(0, K):
                sum_alpha_old += old_alpha[k]

            for k in range(0, K):
                sum_g = 0.0
                sum_h = 0.0

                for j in range(0, nSample):
                    sum_g += scp.digamma(data[k][j] + old_alpha[k])
                    sum_h += scp.digamma(data_row_sum[j] + sum_alpha_old)

                # alpha[k] = mp.mpf(old_alpha[k]*mp.mpf(sum_g - (nSample*self.digamma(old_alpha[k])))/mp.mpf(sum_h - (nSample*self.digamma(sum_alpha_old))))
                alpha[k] = (old_alpha[k] * (sum_g - (nSample * scp.digamma(old_alpha[k]))) / (
                sum_h - (nSample * scp.digamma(sum_alpha_old))))
                self.alpha_temp[k] = alpha[k]

            for j in range(0, K):
                if ((np.fabs(alpha[j]) - old_alpha[j]) > 0.000001):
                    break
                if (j == K - 1):
                    sat_state = True

            if (sat_state == True):
                break

