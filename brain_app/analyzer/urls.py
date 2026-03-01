from django.urls import path
from . import views

app_name = "analyzer"

urlpatterns = [
    path("", views.upload_view, name="upload"),
    path("report/<str:filename>", views.download_report, name="download_report"),
]
