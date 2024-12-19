from django.db import models

# Create your models here.

class Store_pdf(models.Model):
    uploaded_file = models.FileField(upload_to='uploads/',null=True)  # Directory to store files
    uploaded_at = models.DateTimeField(auto_now_add=True)
