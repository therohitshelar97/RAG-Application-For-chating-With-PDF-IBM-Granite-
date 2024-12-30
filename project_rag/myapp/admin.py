from django.contrib import admin
from .models import Store_pdf

# Register your models here.
@admin.register(Store_pdf)
class Pdf_data(admin.ModelAdmin):
    display_list = ['id','uploaded_file']