from django import forms


class MRIUploadForm(forms.Form):
    image = forms.ImageField(
        label="Upload a brain MRI image",
        help_text="Supported formats: PNG, JPEG, TIFF, BMP.",
        widget=forms.ClearableFileInput(attrs={
            "accept": "image/*",
            "class": "form-control",
        }),
    )
