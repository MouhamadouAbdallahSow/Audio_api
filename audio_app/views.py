from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
import os

def remove_noise(audio_file_path):
    # Code pour supprimer le bruit
    return audio_file_path  # Retourne le fichier traité

def remove_echo(audio_file_path):
    # Code pour supprimer l'écho
    return audio_file_path

def remove_clicks(audio_file_path):
    # Code pour supprimer les clics
    return audio_file_path

class AudioUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        if 'audio' not in request.FILES:
            return Response({"error": "Aucun fichier audio fourni."}, status=status.HTTP_400_BAD_REQUEST)

        # Récupérer le fichier uploadé
        audio_file = request.FILES['audio']
        audio_file_path = default_storage.save(f"uploads/{audio_file.name}", audio_file)

        # Chemin complet vers le fichier
        full_audio_path = os.path.join(settings.MEDIA_ROOT, audio_file_path)

        # Appliquer le traitement en fonction des choix
        processing_choice = request.data.get('choice', 'noise')  # Par défaut "noise"
        if processing_choice == 'noise':
            processed_path = remove_noise(full_audio_path)
        elif processing_choice == 'echo':
            processed_path = remove_echo(full_audio_path)
        elif processing_choice == 'clicks':
            processed_path = remove_clicks(full_audio_path)
        else:
            return Response({"error": "Choix de traitement invalide."}, status=status.HTTP_400_BAD_REQUEST)

        # Retourner le fichier traité
        processed_file_url = f"{settings.MEDIA_URL}{os.path.basename(processed_path)}"
        return Response({"message": "Fichier traité avec succès.", "file_url": processed_file_url}, status=status.HTTP_200_OK)
