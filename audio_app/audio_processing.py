# audio_app/audio_processing.py

# import librosa 
# import soundfile as sf

def remove_noise(input_file, output_file):
    """Supprime le bruit d'un fichier audio."""
    y, sr = librosa.load(input_file, sr=None) 
    y_denoised = librosa.effects.preemphasis(y) 
    sf.write(output_file, y_denoised, sr) 
def remove_echo(input_file, output_file):
    """Supprime l'écho d'un fichier audio."""
    pass

def remove_clicks(input_file, output_file):
    """Supprime les clics d'un fichier audio."""

    pass
