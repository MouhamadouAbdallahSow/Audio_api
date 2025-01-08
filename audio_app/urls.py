from django.urls import path
from .views import AudioFileUploadView

urlpatterns = [
    path('audio/', AudioFileUploadView.as_view(), name='audio-upload'),
]
