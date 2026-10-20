import os
from smolagents import tool
from dotenv import load_dotenv
# ----------------------------------------------------------------------------
# SECTION 1: Vision & Multimodal
# ----------------------------------------------------------------------------

load_dotenv()


@tool
def vision_tool(prompt: str, image_list: list) -> str:
    """
    Analyzes one or more images using a multimodal model to answer specific questions.
    It processes image content and returns the model's text-based response.

    Args:
        prompt: The user question or task to perform on the images.
        image_list: A list of PIL Image objects to be analyzed.
    """
    import io, base64, os
    from smolagents import OpenAIServerModel
    from PIL import Image 
    
    model = OpenAIServerModel(
        model_id='gemma4:31b-cloud',
        api_base='https://ollama.com/v1',
        api_key=os.getenv('GEMMA_API_KEY'),
        temperature=0.0,
        max_tokens=2048,
    )
    
    payload = [{"type": "text", "text": prompt}]
    for img in image_list:
        try:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    return model([{"role": "user", "content": payload}]).content

# ----------------------------------------------------------------------------
# SECTION 2: YouTube Pipeline (Tout-en-un pour la robustesse)
# ----------------------------------------------------------------------------
@tool
def ask_youtube_full_pipeline(url: str, question: str) -> str:
    """
    Downloads a YouTube video, extracts key frames at regular intervals, and uses a multimodal model to answer questions about the video's content.

    Args:
        url: The YouTube video URL to process.
        question: The specific question to ask about the video content.
    """
    import os, re, tempfile, yt_dlp, imageio
    from PIL import Image
    from smolagents import OpenAIServerModel

    # 1. Extraction ID interne
    patterns = [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"]
    video_id = None
    
    for pat in patterns:
        match = re.search(pat, url)
        if match:
            video_id = match.group(1)
            break

    if not video_id: 
        return "Erreur: URL invalide."

    with tempfile.TemporaryDirectory() as tmpdir:
        # Configuration optimisée : légère (évite les timeouts) et robuste
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Récupère un fichier MP4 direct déjà fusionné
            'outtmpl': os.path.join(tmpdir, 'downloaded_video.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            
        # CORRECTIF CRITIQUE : On liste le dossier pour trouver le fichier, peu importe son extension exacte
        downloaded_files = os.listdir(tmpdir)
        if not downloaded_files:
            return "Erreur: Le téléchargement de la vidéo a échoué."
            
        video_path = os.path.join(tmpdir, downloaded_files[0])
            
        # Extraction des images via le chemin dynamique détecté
        reader = imageio.get_reader(video_path, format='ffmpeg')
        frames = [Image.fromarray(f) for i, f in enumerate(reader) if i % 60 == 0][:8] 
        reader.close()

        # 2. Analyse VLM (Auto-contenu)
        model = OpenAIServerModel(model_id='gemma4:31b-cloud', api_base='https://ollama.com/v1', api_key=os.getenv('GEMMA_API_KEY'))
        import base64, io
        payload = [{"type": "text", "text": question}]
        for img in frames:
            b = io.BytesIO()
            img.save(b, format="JPEG")
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}"}})
        
        return model([{"role": "user", "content": payload}]).content
# ----------------------------------------------------------------------------
# SECTION 3: Fichiers & Audio
# ----------------------------------------------------------------------------
@tool
def read_pdf_file(file_path: str) -> str:
    """
    Reads the content of a PDF file, extracting all visible text page by page.

    Args:
        file_path: The absolute or relative path to the .pdf file that needs to be read.
    """
    import os
    from pypdf import PdfReader
    
    if not os.path.exists(file_path):
        return f"Erreur : Le fichier au chemin '{file_path}' est introuvable."
        
    try:
        reader = PdfReader(file_path)
        full_text = []
        
        # Parcourir et extraire le texte de chaque page
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                full_text.append(f"\n--- [Page {page_num}] ---")
                full_text.append(page_text.strip())
                
        output = "\n".join(full_text)
        return output if output.strip() else "Le document PDF semble vide ou ne contient que des images (scan)."

    except Exception as e:
        return f"Erreur lors de la lecture du fichier PDF : {str(e)}"

@tool
def read_docx_file(file_path: str) -> str:
    """
    Reads the content of a Word document (.docx file), extracting all text from paragraphs and tables.

    Args:
        file_path: The path to the .docx file that needs to be read.
    """
    import docx
    import os
    
    if not os.path.exists(file_path):
        return f"Erreur : Le fichier au chemin '{file_path}' est introuvable."
        
    try:
        doc = docx.Document(file_path)
        full_text = []

        # 1. Extraction du texte des paragraphes classiques
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        # 2. Extraction du texte des tableaux (très fréquent dans GAIA)
        for table in doc.tables:
            full_text.append("\n--- [Tableau détecté] ---")
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    # On sépare les colonnes par des barres verticales pour l'agent
                    full_text.append(" | ".join(row_text))
            full_text.append("-------------------------\n")

        # Renvoyer le tout proprement assemblé
        output = "\n".join(full_text)
        return output if output.strip() else "Le document est vide."

    except Exception as e:
        return f"Erreur lors de la lecture du fichier .docx : {str(e)}"


@tool
def file_from_url(url: str, save_as: str = "downloaded_file") -> str:
    """
    Downloads a file from a provided URL, determines its extension properly, and saves it.

    Args:
        url: The source URL of the file to download.
        save_as: Optional filename to save the file as. If not provided, derives the name from the URL or headers.
    """
    import requests
    import os
    from urllib.parse import urlparse
    import mimetypes

    # 1. Utiliser un répertoire local dédié et contrôlé (comme ton dossier downloads)
    download_dir = os.path.abspath("downloads")
    os.makedirs(download_dir, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()

            # 2. Détermination intelligente du nom et de l'extension
            if save_as == "downloaded_file":
                base_name = os.path.basename(urlparse(url).path)
                if not base_name or "." not in base_name:
                    # Si l'URL est opaque, on regarde le Content-Type envoyé par le serveur
                    content_type = r.headers.get('content-type', '').split(';')[0]
                    ext = mimetypes.guess_extension(content_type) or ".txt"
                    base_name = f"downloaded_media_{int(os.path.getctime(download_dir))}{ext}"
                save_as = base_name

            file_path = os.path.join(download_dir, save_as)

            # 3. Écriture par blocs (chunks)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Renvoie le chemin absolu impeccable pour tes autres sous-agents
            return os.path.abspath(file_path)

    except Exception as e:
        return f"Erreur lors du téléchargement du fichier depuis l'URL : {str(e)}"

@tool
def transcribe_youtube(yt_url: str) -> str:
    """
    Downloads the audio from a YouTube video and performs speech-to-text transcription using the Groq cloud API (Whisper).
    Useful for retrieving text content or answering questions based on video dialogue without local heavy models.

    Args:
        yt_url: The URL of the YouTube video to transcribe.
    """
    import os, tempfile, yt_dlp, requests
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Erreur : La variable d'environnement GROQ_API_KEY n'est pas configurée dans l'environnement."

    with tempfile.TemporaryDirectory() as tmpdir:
        # Téléchargement de l'audio uniquement
        ydl_opts = {
            "format": "bestaudio", 
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "quiet": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}]
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
                ydl.extract_info(yt_url, download=True)
            
            audio_path = os.path.join(tmpdir, "audio.mp3")
            
            # Appel API Groq Cloud (Whisper Large V3)
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {groq_api_key}"}
            
            with open(audio_path, 'rb') as f:
                files = {"file": f}
                data = {"model": "whisper-large-v3"}
                response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                return f"Erreur API Groq : {response.text}"
                
            return response.json().get("text", "")
            
        except Exception as e:
            return f"Erreur de traitement audio / API : {str(e)}"

@tool
def read_text_file(file_path: str) -> str:
    """
    Reads and returns the raw plain text content from a specified local file path.

    Args:
        file_path: The full local path to the text file.
    """
    with open(file_path, "r", encoding="utf-8") as f: return f.read()

# ----------------------------------------------------------------------------
# SECTION 4: OCR & Tabulaire
# ----------------------------------------------------------------------------
@tool
def extract_text_via_ocr(image_path: str) -> str:
    """
    Extracts text from an image using the cloud-based OCR.space API.
    Returns the detected text found within the provided image path without heavy local dependencies.

    Args:
        image_path: The local path to the image file containing text.
    """
    import requests, os
    
    # Utilisation de la clé publique 'helloworld' d'OCR.space si aucune clé n'est configurée
    api_key = os.getenv("OCR_SPACE_API_KEY", "helloworld")
    
    try:
        with open(image_path, 'rb') as f:
            response = requests.post(
                'https://api.ocr.space/parse/image',
                files={'file': f},
                data={'apikey': api_key, 'language': 'fra'}
            )
        
        result = response.json()
        if result.get("IsErroredOnProcessing"):
            return f"Erreur OCR.space : {result.get('ErrorMessage')}"
        
        parsed_results = result.get('ParsedResults', [])
        if not parsed_results:
            return "Aucun texte détecté sur l'image."
            
        return parsed_results[0].get('ParsedText', '')
        
    except Exception as e:
        return f"Erreur lors de l'appel API OCR : {str(e)}"

@tool
def summarize_csv_data(path: str, query: str = "") -> str:
    """
    Loads a CSV file into a pandas DataFrame and generates a descriptive summary, including column statistics.
    Optionally allows filtering rows via a pandas query string.

    Args:
        path: The local file path to the CSV file.
        query: Optional pandas query expression to filter rows before summarizing (e.g., "age > 30").
    """
    import pandas as pd
    df = pd.read_csv(path)
    if query: df = df.query(query)
    return df.describe().to_string()

@tool
def calculer_probabilite_loi(loi: str, parametres: dict, valeur_k: float) -> str:
    """
    Calculates the exact or cumulative probability for a specified statistical distribution law (Normal or Binomial).

    Args:
        loi: The distribution type, either 'normale' or 'binomiale'.
        parametres: Dictionary containing distribution parameters (e.g., {'moyenne': float, 'ecart_type': float} for Normal).
        valeur_k: The value or bound for which the cumulative probability P(X <= k) is calculated.
    """
    # Import local obligatoire pour l'autonomie en sandbox
    from scipy import stats
    
    try:
        if loi.lower() == "normale":
            loc = parametres.get("moyenne", 0)
            scale = parametres.get("ecart_type", 1)
            prob = stats.norm.cdf(valeur_k, loc=loc, scale=scale)
            return f"Pour une Loi Normale({loc}, {scale}), P(X <= {valeur_k}) = {prob:.6f}"

        elif loi.lower() == "binomiale":
            n = int(parametres.get("n", 1))
            p = float(parametres.get("p", 0.5))
            prob = stats.binom.cdf(valeur_k, n, p)
            prob_exacte = stats.binom.pmf(valeur_k, n, p)
            return (
                f"Pour une Loi Binomiale(n={n}, p={p}) :\n"
                f"- Probabilité exacte P(X = {valeur_k}) = {prob_exacte:.6f}\n"
                f"- Probabilité cumulative P(X <= {valeur_k}) = {prob:.6f}"
            )
        else:
            return "Loi non supportée. Utilisez 'normale' ou 'binomiale'."
    except Exception as e:
        return f"Erreur de paramètres : {str(e)}"
    
@tool
def summarize_excel_data(path: str, query: str = "") -> str:
    """
    Loads an Excel file (.xls or .xlsx) into a pandas DataFrame and generates a descriptive summary, including column statistics.
    Optionally allows filtering rows via a pandas query string.

    Args:
        path: The local file path to the Excel file.
        query: Optional pandas query expression to filter rows before summarizing (e.g., "age > 30").
    """
    import pandas as pd
    df = pd.read_excel(path)
    if query: df = df.query(query)
    return df.describe().to_string()

@tool
def analyser_serie_statistique(donnees: list[float]) -> str:
    """
    Calculates essential descriptive statistics for a provided series of numerical data, such as mean, median, standard deviation, and range.

    Args:
        donnees: A list of floats representing the dataset to analyze.
    """
    import numpy as np
    arr = np.array(donnees)
    return f"Moyenne: {np.mean(arr):.2f}, Écart-type: {np.std(arr):.2f}, Min: {np.min(arr)}, Max: {np.max(arr)}"


# ----------------------------------------------------------------------------
# SECTION 5: Maths & Chess
# ----------------------------------------------------------------------------
@tool
def resoudre_calcul_formel(expression: str, action: str, variable: str = "x") -> str:
    """
    Performs formal mathematical computations such as solving equations, differentiation, integration, and simplification using sympy.

    Args:
        expression: The mathematical expression in Python syntax (e.g., "x**2 + 2*x - 3" or "x**2 - 4 = 0").
        action: The mathematical operation to perform ('resoudre', 'deriver', 'integrer', or 'simplifier').
        variable: The primary variable in the expression to operate on. Defaults to 'x'.
    """
    import sympy as sp
    v = sp.Symbol(variable)
    e = sp.sympify(expression.split("=")[0]) if "=" in expression else sp.sympify(expression)
    if action == "simplifier": return str(sp.simplify(e))
    if action == "deriver": return str(sp.diff(e, v))
    if action == "integrer": return str(sp.integrate(e, v))
    return str(sp.solve(e, v))

@tool
def analyze_chess_position(fen: str) -> str:
    """
    Analyzes a chess board state from a FEN (Forsyth-Edwards Notation) string.
    Detects checks, checkmates, stalemates, lists legal moves, and scans for immediate 'Mate in 1' opportunities.

    Args:
        fen: The FEN string representing the current board state.
    """
    import chess
    board = chess.Board(fen)
    moves = [board.san(m) for m in board.legal_moves]
    return f"Turn: {'White' if board.turn else 'Black'}. Moves: {', '.join(moves)}"

@tool
def get_stockfish_evaluation(fen: str) -> str:
    """
    Calls an external Chess API to get a deep engine evaluation (Stockfish 18) for a given FEN string.
    It returns the absolute best move, tactical score, winning chances, and the expected continuation line.

    Args:
        fen: The FEN string representing the current board state to analyze.
    """
    import requests
    r = requests.post("https://chess-api.com/v1", json={"fen": fen, "depth": 12})
    d = r.json()
    return f"Best: {d.get('san')}, Eval: {d.get('eval')}"