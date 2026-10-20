import requests
import os
import shutil
from utils.logger import get_logger
from smolagents import Tool

logger = get_logger(__name__)

class GetTaskFileTool(Tool):
    name = "get_task_file_tool"
    description = "If a file_name is provided, download the file associated with a given task_id and return its absolute local path."
    inputs = {
        "task_id": {"type": "string", "description": "The unique task ID (required)"},
        "file_name": {"type": "string", "description": "The exact name of the file to download or copy (required)"},
    }
    output_type = "string"

    def __init__(self, **kwargs):
        # CORRECTIF 1: Appel obligatoire de l'initialisation parente pour smolagents
        super().__init__(**kwargs)
        self.evaluation_api_base_url = "https://agents-course-unit4-scoring.hf.space"
        self.directory_name = "downloads"
        self.create_dir()
        
    def forward(self, task_id: str, file_name: str) -> str:
        target_path = os.path.join(self.directory_name, file_name)
        try:
            response = requests.get(f"{self.evaluation_api_base_url}/files/{task_id}", timeout=15)
            response.raise_for_status()
            with open(target_path, 'wb') as file:
                file.write(response.content)
            return os.path.abspath(target_path)
        except Exception as e:
            logger.warning(f"Échec du téléchargement pour la tâche {task_id}, tentative de récupération locale : {e}")
            
            # Source locale présumée (ex: validation/ ou files/)
            local_source = f"files/{file_name}"
            if not os.path.exists(local_source):
                # Fallback au cas où le dataset GAIA utilise l'arborescence complète
                local_source = file_name 
                
            try:
                shutil.copy2(local_source, target_path)
                # CORRECTIF 2: Renvoie TOUJOURS le chemin absolu, même en mode secours
                return os.path.abspath(target_path)
            except Exception as copy_error:
                return f"Erreur critique : Impossible de trouver le fichier local ou distant ({copy_error})"
        
    def create_dir(self):
        if not os.path.exists(self.directory_name):
            os.makedirs(self.directory_name)
            logger.info(f"Directory '{self.directory_name}' created successfully.")
        else:
            logger.debug(f"Directory '{self.directory_name}' already exists.")