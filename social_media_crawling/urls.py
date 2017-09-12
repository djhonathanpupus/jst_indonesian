from django.conf.urls import url
from . import views
from .models import TwitterCrawl, Crawl
from django.views.generic import ListView

class listData(ListView):
    models = Crawl
    queryset = Crawl.objects.all()
    

    
urlpatterns = [
    url(r'^$', views.index, name='index'),
#     url(r'^search/$', views.crawl, name = 'crawl'),
#     url(r'^hasil/$', ListView.as_view(queryset=Crawl.objects.all(), template_name = "WMMS/db_show.html"), name = 'hasil'),
    url(r'^hasil2/$', ListView.as_view(queryset=TwitterCrawl.objects.all(), template_name = "WMMS/db_show2.html"), name = 'hasil2'),
    url(r'^del/$', views.delet),
    url(r'^del2/$', views.delet2),
    url(r'^search2/$', views.crawl2, name = 'crawl2'),
#     url(r'^test/', views.fileUpload),
#     url(r'^test1/', views.analisis1),
#     url(r'^search2/download', views.download),
    url(r'search2/visualisasi/$', views.visualisasi),
    url(r'search2/visualisasi1/$', views.visualisasi2),
    url(r'search2/saveDB/$', views.savetoDB, name="savetoDB"),
    url(r'manage_facebook/$', views.fullDBFB, name="manageFB"),
    url(r'manage_twitter/$', views.fullDBTW, name="manageTW")
    ]
