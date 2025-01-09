import os
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import FileResponse

# Import des fonctions de traitement
from processing_audio import remove_noise, remove_echo, remove_clicks


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

    def get(self, request, *args, **kwargs):
        """
        Méthode GET pour renvoyer un fichier traité.
        Le fichier doit être spécifié par le paramètre GET `file` (e.g., ?file=processed_audio.mp3).
        """
        # Récupérer le nom du fichier traité à partir du paramètre GET
        file_name = request.GET.get('file')
        if not file_name:
            return Response({"error": "Veuillez spécifier le nom du fichier en tant que paramètre GET (file)."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Construire le chemin vers le fichier traité
        file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', file_name)

        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            return Response({"error": "Le fichier demandé n'existe pas."}, status=status.HTTP_404_NOT_FOUND)

        # Ouvrir le fichier et le renvoyer dans la réponse
        try:
            file = open(file_path, 'rb')
            response = FileResponse(file, content_type="audio/mpeg")
            response['Content-Disposition'] = f'attachment; filename="{file_name}"'
            return response
        except Exception as e:
            return Response({"error": f"Erreur lors de l'envoi du fichier : {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
