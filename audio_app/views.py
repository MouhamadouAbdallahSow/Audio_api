import os
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import FileResponse
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


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
            processed_path = reduce_noise(full_audio_path)
        elif processing_choice == 'replace_silence':
            processed_path = detect_and_replace_silences(full_audio_path)
        elif processing_choice == 'reduce_silence':
            processed_path = detect_and_reduce_silences(full_audio_path)
        elif processing_choice == 'clicks':
            processed_path = remove_clicks(full_audio_path)
        else:
            return Response({"error": "Choix de traitement invalide."}, status=status.HTTP_400_BAD_REQUEST)

        processed_file_url = f"{settings.MEDIA_URL}{os.path.basename(processed_path)}"
        return Response({"message": "Fichier traité avec succès.", "file_url": processed_file_url}, status=status.HTTP_200_OK)

    def get(self, request, *args, **kwargs):

        file_name = request.GET.get('file')
        if not file_name:
            return Response({"error": "Veuillez spécifier le nom du fichier en tant que paramètre GET (file)."},
                            status=status.HTTP_400_BAD_REQUEST)

        file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', file_name)

        if not os.path.exists(file_path):
            return Response({"error": "Le fichier demandé n'existe pas."}, status=status.HTTP_404_NOT_FOUND)

        try:
            file = open(file_path, 'rb')
            response = FileResponse(file, content_type="audio/mpeg")
            response['Content-Disposition'] = f'attachment; filename="{file_name}"'
            return response
        except Exception as e:
            return Response({"error": f"Erreur lors de l'envoi du fichier : {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def detect_and_replace_silences(file_path, threshold=0.01, frame_duration=0.02):

    fs, signal = read(file_path)
    
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    
    signal = signal / np.max(np.abs(signal))
    
    frame_size = int(fs * frame_duration)  
    num_frames = len(signal) // frame_size  
    energy_threshold = threshold  
    
    energies = np.array([
        np.sum(signal[i * frame_size:(i + 1) * frame_size] ** 2)
        for i in range(num_frames)
    ])
    energies = energies / np.max(energies)  
    
    active_frames = np.where(energies > energy_threshold)[0]
    silence_frames = np.where(energies <= energy_threshold)[0]
    
    modified_signal = []
    for i in range(num_frames):
        frame = signal[i * frame_size:(i + 1) * frame_size]
        if i in active_frames:
            modified_signal.append(frame)  
        else:
            modified_signal.append(np.zeros_like(frame))  
    
    modified_signal = np.concatenate(modified_signal)
    
    modified_signal = (modified_signal * 32767).astype(np.int16)
    
    write(file_path, fs, modified_signal)  
    return file_path
    

def detect_and_reduce_silences(file_path, threshold=0.01, frame_duration=0.02, attenuation_factor=0.1):
    fs, signal = read(file_path)
    
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    
    signal = signal / np.max(np.abs(signal))
    
    frame_size = int(fs * frame_duration)  
    num_frames = len(signal) // frame_size  
    energy_threshold = threshold  
    energies = np.array([
        np.sum(signal[i * frame_size:(i + 1) * frame_size] ** 2)
        for i in range(num_frames)
    ])
    energies = energies / np.max(energies)  
    
    modified_signal = np.zeros_like(signal)
    
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        
        if energies[i] > energy_threshold:
            modified_signal[start:end] = signal[start:end]
        else:
            modified_signal[start:end] = signal[start:end] * attenuation_factor
    
    modified_signal = (modified_signal * 32767).astype(np.int16)
    
    write(file_path, fs, modified_signal) 
    return file_path
    

def reduce_noise(input_file: str, noise_start: float = 0, noise_end: float = 1):
    audio, sr = librosa.load(input_file, sr=None)
    
    noise_sample = audio[int(noise_start * sr): int(noise_end * sr)]
    
    noise_spectrum = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)
    
    audio_stft = librosa.stft(audio)
    audio_magnitude, audio_phase = np.abs(audio_stft), np.angle(audio_stft)
    
    noise_reduced_magnitude = np.maximum(audio_magnitude - noise_spectrum[:, np.newaxis], 0)
    
    audio_denoised_stft = noise_reduced_magnitude * np.exp(1j * audio_phase)
    audio_denoised = librosa.istft(audio_denoised_stft)
    
    sf.write(input_file, audio_denoised, sr)
    return input_file
    
    

def remove_clicks(file_path, threshold_factor=10000, window_size=50):

    audio, sr = librosa.load(file_path, sr=None)
    
    # Calculer le RMS pour déterminer le seuil
    rms_value = np.sqrt(np.mean(audio**2))
    threshold = rms_value / threshold_factor
    
    # Détection des indices des clics
    click_indices = np.where(np.abs(audio) > threshold)[0]
    
    # Réparer les clics
    repaired_audio = np.copy(audio)
    for idx in click_indices:
        # Définir une fenêtre autour du clic
        start = max(0, idx - window_size)
        end = min(len(audio), idx + window_size)
        
        # Interpoler les valeurs
        if start > 0 and end < len(audio):
            repaired_audio[idx] = np.mean(np.concatenate((audio[start:idx], audio[idx+1:end])))
    
    sf.write(file_path, repaired_audio, sr)
    return file_path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# def supprimer_silences(audio_path, seuil_silence=-40, min_duree_silence=1000, output_path="audio_sans_silence.wav"):

#     try:
#         # Charger le fichier audio
#         audio = AudioSegment.from_file(audio_path)
#     except Exception as e:
#         raise ValueError(f"Erreur lors du chargement du fichier audio : {e}")

#     # Détecter les segments non silencieux
#     non_silent_segments = detect_nonsilent(audio, min_silence_len=min_duree_silence, silence_thresh=seuil_silence)

#     if not non_silent_segments:
#         raise ValueError("Aucun segment non silencieux détecté.")

#     # Créer un nouvel audio en concaténant les segments non silencieux
#     audio_sans_silence = AudioSegment.empty()
#     for start, end in non_silent_segments:
#         audio_sans_silence += audio[start:end]

#     # Exporter l'audio résultant
#     try:
#         audio_sans_silence.export(output_path, format="wav")
#     except Exception as e:
#         raise ValueError(f"Erreur lors de l'exportation du fichier audio : {e}")

#     return output_path



# def bandpass_filter(data, sr, lowcut, highcut, order=5):

#     nyquist = 0.5 * sr  # Fréquence de Nyquist
#     low = lowcut / nyquist
#     high = highcut / nyquist

#     # Conception du filtre Butterworth passe-bande
#     b, a = butter(order, [low, high], btype='band')
#     filtered_data = lfilter(b, a, data)
#     return filtered_data

# def process_audio(input_file, output_file, lowcut=300, highcut=3400, order=5):

#     audio, sr = librosa.load(input_file, sr=None)
    
#     # Appliquer le filtre passe-bande
#     filtered_audio = bandpass_filter(audio, sr, lowcut, highcut, order)

#     # Sauvegarder l'audio filtré
#     sf.write(output_file, filtered_audio, sr)
#     return output_file

