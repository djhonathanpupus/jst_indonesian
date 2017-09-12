from django import forms
from django.forms.widgets import DateInput
from bokeh.models.widgets import widget


lang = [('id','indonesia'),('en','english'),('ja','Japanese'),('ar','arabic'),('es','Spanyol'),('am','amharic')]
##Fix form
CHOICE = [('0','NO'),('1','YES')]
class upFile(forms.Form):
    file = forms.FileField()

class fixbrowse(forms.Form):
    content = forms.CharField(max_length=500)
    language = forms.ChoiceField(choices=lang,widget=forms.Select(attrs={"class": "selectpicker"}))

class fixscrape(forms.Form):
    tapdown = forms.IntegerField(min_value=10, widget=forms.TextInput(attrs={}))
    date = forms.ChoiceField(label= '',choices=CHOICE, widget=forms.RadioSelect(attrs={'onchange': 'forma("datepick")'}))
    
# class loginForms(forms.Form):
#     username = forms.CharField(max_length=100)
#     password = forms.CharField(widget = forms.PasswordInput)
    