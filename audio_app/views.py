from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .models import AudioFile
from .serializers import AudioFileSerializer
from .audio_processing import remove_noise, remove_echo, remove_clicks
import os

class AudioFileUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        # Sérialiser les données d'upload du fichier audio
        serializer = AudioFileSerializer(data=request.data)
        if serializer.is_valid():
            # Sauvegarder le fichier audio dans la base de données
            audio_file = serializer.save()

            # Chemins des fichiers audio
            input_file = audio_file.audio.path  # Chemin du fichier audio uploadé
            output_file = os.path.splitext(input_file)[0] + '_processed.wav'  # Chemin pour le fichier traité

            # Récupérer les traitements à appliquer depuis la requête
            process_choices = request.data.get('process', [])  # Liste des traitements

            # Appliquer les traitements audio en fonction de la liste choisie
            if 'noise' in process_choices:
                remove_noise(input_file, output_file)  # Traitement de suppression du bruit
            if 'echo' in process_choices:
                remove_echo(output_file, output_file)  # Traitement de suppression des échos
            if 'clicks' in process_choices:
                remove_clicks(output_file, output_file)  # Traitement de suppression des clics

            # Optionnel: Remplacer le fichier original par le fichier traité (facultatif)
            audio_file.audio.name = os.path.basename(output_file)
            audio_file.save()

            # Retourner la réponse avec les informations du fichier traité
            return Response({
                "id": audio_file.id,
                "original_file": audio_file.audio.url,  # URL du fichier original
                "processed_file": f"/media/audio_files/{os.path.basename(output_file)}"  # URL du fichier traité
            }, status=status.HTTP_201_CREATED)

        # Si le fichier n'est pas valide, retourner une erreur
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, *args, **kwargs):
        # Récupérer tous les fichiers audio dans la base de données
        audio_files = AudioFile.objects.all()
        # Sérialiser les fichiers audio
        serializer = AudioFileSerializer(audio_files, many=True)
        # Retourner la réponse avec tous les fichiers audio
        return Response(serializer.data)
