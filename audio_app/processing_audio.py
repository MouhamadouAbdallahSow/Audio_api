import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import librosa
import soundfile as sf


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


def remove_clicks(audio, sr, threshold_factor=10000, window_size=50):

    rms_value = np.sqrt(np.mean(audio**2))
    threshold = rms_value / threshold_factor  # Ajuste ce seuil selon ton signal
    click_indices = np.where(np.abs(audio) > threshold)[0]

    # Supprimer les clics
    repaired_audio = np.copy(audio)
    for idx in click_indices:
        # Définir une fenêtre autour du clic
        start = max(0, idx - window_size)
        end = min(len(audio), idx + window_size)
        
        # Interpoler les valeurs
        if start > 0 and end < len(audio):
            repaired_audio[idx] = np.mean(np.concatenate((audio[start:idx], audio[idx+1:end])))
    
    return repaired_audio

