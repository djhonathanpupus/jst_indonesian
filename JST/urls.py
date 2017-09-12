from django.conf.urls import url
from . import views

app_name = 'JST'
urlpatterns = [
    url(r'^$', views.halamanMuka, name='halamanMuka'),
    url(r'Sentimen/inputData$', views.inputDataSentimen, name='inputDataSentimen'),
    url(r'Sentimen/simpanSentimen$', views.simpanSentimen, name='simpanSentimen'),
    url(r'Stopwords/simpanStopwords$', views.simpanStopwords, name='simpanStopwords'),
    url(r'previewMI$', views.previewMI, name='previewMI'),
    url(r'prosesMI$', views.prosesMI, name='prosesMI'),
    url(r'Formalisasi/inputData/$', views.inputDataFormalisasi, name='inputDataFormalisasi'),
    url(r'Formalisasi/simpanData/$', views.simpanFormalisasiKata, name='simpanFormalisasiKata'),
    url(r'Pelabelan/previewPelabelan$', views.inputDataPelabelan, name='inputDataPelabelan'),
    url(r'Pelabelan/simpanPelabelan$', views.simpanPelabelan, name='simpanPelabelan'),
    url(r'validasi$', views.validasi, name='validasi'),
]