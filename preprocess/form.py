from django import forms

metode = [('EDR','Edit Distance + Rule'),('ED','Edit Distance'),('BG','Bigram')]
# metode1 = [('FR', 'Formalisasi')]
class PostForm(forms.Form):
    keyword = forms.CharField(max_length=256)
    jumlah =  forms.IntegerField()
    method = forms.ChoiceField(choices=metode, widget=forms.RadioSelect(attrs={'class': 'selectpicker'}))
#     method1 = forms.ChoiceField(choices=metode1, widget=forms.CheckboxInput(attrs={'class': 'selectpicker'}))
