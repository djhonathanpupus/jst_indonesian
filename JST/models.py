from django.db import models
from django.utils import timezone

# Create your models here.
#Dari Fungsi Formalisasi
class FormalisasiKataDB(models.Model):
    kataInformal = models.CharField(max_length=30, primary_key=True)
    kataFormal = models.CharField(max_length=30)
    date = models.DateTimeField(default=timezone.now, blank=True)

class KataFormalDB(models.Model):
    kataFormal = models.CharField(max_length=30, primary_key=True)
    date = models.DateTimeField(default=timezone.now, blank=True)

#Dari Fungsi Sentimen
class SentimenDB(models.Model):
    kataSentimen = models.CharField(max_length=30, primary_key=True)
    sentiLab = models.SmallIntegerField()
    priorNetral = models.DecimalField(max_digits=3, decimal_places=2)
    priorPositive = models.DecimalField(max_digits=3, decimal_places=2)
    priorNegative = models.DecimalField(max_digits=3, decimal_places=2)
    date = models.DateTimeField(default=timezone.now, blank=True)

#Digunakan sebagai model prior berikutnya
class KataDB(models.Model):
    idKata = models.AutoField(primary_key=True)
    kata = models.CharField(max_length=30)
    sentiLabKata = models.SmallIntegerField()

class SentimenKataMIDB(models.Model):
    kataSentimen = models.CharField(max_length=30, primary_key=True)
    sentiLab = models.SmallIntegerField()
    priorNetral = models.DecimalField(max_digits=3, decimal_places=2)
    priorPositive = models.DecimalField(max_digits=3, decimal_places=2)
    priorNegative = models.DecimalField(max_digits=3, decimal_places=2)
    date = models.DateTimeField(default=timezone.now, blank=True)

class StopwordsIDDB(models.Model):
    kataStopword = models.CharField(max_length=30,primary_key=True)
    date = models.DateTimeField(default=timezone.now, blank=True)