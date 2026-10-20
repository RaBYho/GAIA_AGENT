import re
from smolagents import Tool
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptionTool(Tool):
    """
    Tool to fetch the transcript of a YouTube video given its URL.
    Attempts to retrieve official or auto-generated subtitles instantly.

    Args:
        video_url (str): YouTube video URL.

    Returns:
        str: Transcript of the video as a single string.
    """

    name = "youtube_transcription"
    description = "Fetches the transcript of a YouTube video given its URL"
    inputs = {
        "video_url": {"type": "string", "description": "YouTube video URL"},
    }
    output_type = "string"

    def _extract_video_id(self, url: str) -> str:
        """
        Extracts the 11-character YouTube video ID from various URL formats.
        Gathers classic watch, shared (youtu.be), embeds, and shorts formats.
        """
        pattern = r'(?:v=|\/embed\/|\/shorts\/|\/|be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        # Fallback basique si l'expression régulière échoue
        return url.strip().split("v=")[-1].split("&")[0]

    def forward(self, video_url: str) -> str:
        try:
            video_id = self._extract_video_id(video_url)
            
            # Récupération du transcript (tente de récupérer l'anglais et le français par défaut)
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'fr'])
            
            return " ".join([entry["text"] for entry in transcript])
            
        except Exception as e:
            # On renvoie l'erreur sous forme de chaîne pour que l'agent comprenne qu'il doit 
            # basculer sur un plan B (comme ton outil de téléchargement audio + faster-whisper)
            return (
                f"Error fetching transcript via API: {str(e)}. "
                "The video might not have subtitles enabled, or the API is rate-limited. "
                "Please try downloading the audio and transcribing it instead."
            )