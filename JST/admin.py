from django.contrib import admin
from .models import SentimenDB, FormalisasiKataDB, KataDB, KataFormalDB, SentimenKataMIDB, StopwordsIDDB

# Register your models here.
class SentimenDBAdmin(admin.ModelAdmin):
    list_display = ('kataSentimen', 'sentiLab', 'priorPositive',
                    'priorNegative', 'priorNetral', 'date'
                    )
    search_fields = ('kataSentimen', 'sentiLab')
    ordering = ['kataSentimen','sentiLab', 'date']

class FormalisasiKataDBAdmin(admin.ModelAdmin):
    list_display = ('kataInformal', 'kataFormal', 'date')
    search_fields = ('kataInformal', 'kataFormal')
    ordering = ['kataInformal', 'date']

class KataDBAdmin(admin.ModelAdmin):
    list_display = ('idKata', 'kata', 'sentiLabKata')
    search_fields = ('kata', 'idKata')
    ordering = ['idKata', 'kata']

class KataFormalDBAdmin(admin.ModelAdmin):
    list_display = ('kataFormal', 'date')
    search_fields = ('kataFormal', 'date')
    ordering = ['kataFormal', 'date']

class SentimenKataMIDBAdmin(admin.ModelAdmin):
    list_display = ('kataSentimen', 'sentiLab', 'priorPositive',
                    'priorNegative', 'priorNetral', 'date'
                    )
    search_fields = ('kataSentimen', 'sentiLab')
    ordering = ['kataSentimen', 'sentiLab', 'date']

class StopwordsIDDBAdmin(admin.ModelAdmin):
    list_display = ('kataStopword', 'date')
    search_fields = ('kataStopword', 'date')
    ordering = ['kataStopword', 'date']

admin.site.register(SentimenDB, SentimenDBAdmin)
admin.site.register(FormalisasiKataDB, FormalisasiKataDBAdmin)
admin.site.register(KataDB, KataDBAdmin)
admin.site.register(KataFormalDB, KataFormalDBAdmin)
admin.site.register(SentimenKataMIDB, SentimenKataMIDBAdmin)
admin.site.register(StopwordsIDDB, StopwordsIDDBAdmin)
