from django.db import models


# Create your models here.
class Crawl(models.Model):
    name = models.CharField(max_length=100)
    content = models.CharField(max_length=140)
    c_at = models.DateField()
    #Retweet = models.CharField(max_length=100)
    #hashtag = models.CharField(max_length=100)
    
    def _str_(self):
        return  self.content, self.name

class TwitterCrawl(models.Model):
    name = models.CharField(max_length=100)
    tweet = models.CharField(max_length=140)
    date = models.DateField()
    Retweet_user = models.CharField(max_length=100, null=True)
    hashtag = models.CharField(max_length=100, null = True)
    
    def _str_(self):
        return  self.tweet, self.name,  self.Retweet_user, self.hashtag
    
class TwitterTopik(models.Model):
    topik =  models.CharField(max_length = 150) 
    
    def _str_(self):
        return   self.topik
    
class TwitterDataset(models.Model):
    name = models.CharField(max_length=100)
    tweet = models.CharField(max_length=140)
    date = models.DateField()
    Retweet_user = models.CharField(max_length=100, null=True)
    hashtag = models.CharField(max_length=100, null = True)
    topik = models.ForeignKey(TwitterTopik, on_delete=models.CASCADE)
    
    def _str_(self):
        return  self.tweet, self.name,  self.Retweet_user, self.hashtag

class FacebookCrawl(models.Model):
    name = models.CharField(max_length=100)
    status = models.CharField(max_length=5000)
    like = models.IntegerField(blank=True)
    comment = models.IntegerField(blank=True)
    share = models.IntegerField(blank=True)
    
    def _str_(self):
        return   self.name, self.status, int(self.like), int(self.comment), int(self.share)

class FacebookTopik(models.Model):    
    topik = models.CharField(max_length=150)
    
#     user = 
class FacebookDataset(models.Model):
    name = models.CharField(max_length=100)
    status = models.CharField(max_length=5000)
    like = models.IntegerField(blank=True)
    comment = models.IntegerField(blank=True)
    share = models.IntegerField(blank=True)
    topik = models.ForeignKey(FacebookTopik, on_delete=models.CASCADE)
    
    def _str_(self):
        return   self.name, self.status, int(self.like), int(self.comment), int(self.share)

