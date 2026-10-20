import os
from smolagents import Tool
from faster_whisper import WhisperModel


class OpenAISpeechToTextTool(Tool):
    """
    Tool to convert speech to text using the high-performance local faster-whisper model.

    Args:
        audio_path (str): Path to the local audio file (.mp3, .wav, .m4a, etc.).

    Returns:
        str: Transcribed text from the audio file.
    """

    name = "transcribe_audio"
    description = "Transcribes local audio files to text using high-precision faster-whisper."
    inputs = {
        "audio_path": {"type": "string", "description": "Path to the local audio file to transcribe"},
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # On charge le modèle large-v3 une seule fois au démarrage de l'agent.
        # "int8" permet de garder une précision maximale tout en épargnant ta RAM sur CPU.
        # Si tu as un GPU Nvidia disponible, tu peux mettre device="cuda" et compute_type="float16".
        try:
            self.model = WhisperModel("small", device="cpu", compute_type="int8")
        except Exception as e:
            # Sécurité si le chargement échoue au démarrage
            self.model = None
            print(f"Warning: Failed to load faster-whisper model: {e}")

    def forward(self, audio_path: str) -> str:
        if self.model is None:
            return "Error: Whisper model was not initialized properly."

        if not os.path.exists(audio_path):
            return f"Error: Audio file not found at {audio_path}"

        try:
            # Transcription optimisée avec recherche par faisceaux
            segments, info = self.model.transcribe(
                audio_path, 
                beam_size=5,
                best_of=5
            )
            
            # On regroupe les segments de texte décodés
            transcription = [segment.text for segment in segments]
            return " ".join(transcription).strip()

        except Exception as e:
            return f"Error transcribing audio: {str(e)}"