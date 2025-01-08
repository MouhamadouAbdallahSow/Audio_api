from django.db import models

from django.db import models

class AudioFile(models.Model):
    title = models.CharField(max_length=255)
    audio = models.FileField(upload_to='audio_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
