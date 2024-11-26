from django import forms
from .models import FaceEncoding

class FaceEncodingForm(forms.ModelForm):
    class Meta:
        model = FaceEncoding
        fields = ['encoding_file']
