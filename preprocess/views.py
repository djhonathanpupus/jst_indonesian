from django.shortcuts import render
from .ED_rule import correction_3, bigram_corr5, F_EDR
from .ED import correction_2, bigram_corr4, F_ED
from .bigram import bigram_corr3, bigram_corr4, F_BG
from .formalisasi import correction, correction_2
from preprocess.form import PostForm
from preprocess.form2 import PostForm2
from .tests import getall

# Create your views here.
# def index(request):
#     tp = ['Edit Distance + Rule','Formalisasi']
#     input = ''
#     hasil = ''
#     textb = []
#     if request.method=="POST":
#         if 'inputA' in request.POST:
#             PR = request.POST.get("PR")
#             input = request.POST.get('inputtext')
#             hasil = correction_3(input)
#         elif 'inputB' in request.POST:
#             PS = request.POST.get("PS")
#             input = request.POST.get('inputtext')
#             hasil = correction(input)
#     return render(request, 'index_preprocess.html',{'input':input, 'hasil':hasil,'data':tp,'text1':textb})

# def index(request):
#     input = ''
#     hasil = ''
#     textb = []
#     situs1 = request.POST.get('method', '')
#     if situs1=='EDR':
#         input = request.POST.get('inputtext')
#         hasil = correction_3(input)
#     elif situs1 == 'ED':
#         input = request.POST.get('inputtext')
#         hasil = correction_2(input)
#     elif situs1 == 'BG':
#         input = request.POST.get('inputtext')
#         hasil = bigram_corr3(input)
#     elif situs1 == 'FR':
#         input = request.POST.get('inputtext')
#         hasil = correction(input)
#     return render(request, 'index_preprocess.html', {'input':input, 'hasil':hasil, 'f':PostForm})

def index(request):
    input = ''
    hasil = ''
    textb = []
#     hasil1 = bigram_corr5(gettweet())
    situs1 = request.POST.get('method', '')
    if situs1=='EDR':
        input = request.POST.get('inputtext')
        hasil = F_EDR(input)
    elif situs1 == 'ED':
        input = request.POST.get('inputtext')
        hasil = F_ED(input)
    elif situs1 == 'BG':
        input = request.POST.get('inputtext')
        hasil = F_BG(input)
    situs2 = request.POST.get('method1', '')
    if situs2 == 'FR':
        input = request.POST.get('inputtext')
        hasil = correction(input)
#     if 'fromDB' in request.POST:
#         hasil1 = bigram_corr5(gettweet())
    return render(request, 'index_preprocess.html', {'input':input, 'hasil':hasil, 'f':PostForm, 'f2':PostForm2})

def upload(request):
    hasil = ''
    IA = ''
    if request.method=="POST":
        if 'input' in request.POST:            
            IA = request.POST.get("inputArea")
            if IA == "": #kalau menginput dengan file (multi input)
                inputFile = request.FILES["inputDataTest"]
                loadfile = TextIOWrapper(inputFile.file,encoding='utf-8')
                for i in loadfile:
                    hasil = correction(i)
#             datatemp = []
#             for i in loadfile:
#                 datatemp.append(i)
#             for i in datatemp:
#                 prepros, prediction = predict(i,int(FE),topik) #memanggil fungsi predict dari listFunction
#                 tabledata.append({
#                 'input': i,
#                 'prepros': prepros,
#                 'prediction': prediction,
#                 'confirm': True
#                         })
#                 data = json.dumps(tabledata)
    return render(request, "index_preprocess.html",{'IA':IA, 'output':hasil})