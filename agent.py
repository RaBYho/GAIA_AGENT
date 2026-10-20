import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, LocalPythonExecutor
from smolagents import GoogleSearchTool, VisitWebpageTool, FinalAnswerTool
from tools.get_answer import GetTaskFileTool
from tools.tools import (
    vision_tool, 
    read_text_file,
    transcribe_youtube,  
    extract_text_via_ocr, 
    summarize_csv_data,
    summarize_excel_data,
    analyser_serie_statistique,
    calculer_probabilite_loi,
    resoudre_calcul_formel,
    analyze_chess_position,
    get_stockfish_evaluation,
    ask_youtube_full_pipeline,
    read_docx_file,
    read_pdf_file,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# Imports minimaux partagés
BASE_IMPORTS = ["pandas", "numpy", "requests","openpyxl"]

# Imports pour CodeMathAgent
MATH_IMPORTS = BASE_IMPORTS + ["scipy", "sympy"]

# Imports pour ResearchAgent (les plus lourds)
RESEARCH_IMPORTS = BASE_IMPORTS + [
     "imageio", "PIL", "cv2", "yt_dlp","docx","pypdf"
]
# Imports pour ChessAgent (très léger)
CHESS_IMPORTS = ["chess"]

local_research_executor = LocalPythonExecutor(
    additional_authorized_imports=RESEARCH_IMPORTS,
    timeout_seconds=200 # <--- C'est ici qu'on débloque la limite !
)

try :
    
    load_dotenv()
    
    GEMMA_API_KEY = os.getenv('GEMMA_API_KEY')
    GPT_OSS_API_KEY = os.getenv('GPT_OSS_API_KEY')
    E2B_API_KEY = os.getenv('E2B_API_KEY')
    GROK_API_KEY = os.getenv('GROK_API_KEY')
    OCR_SPACE_API_KEY = os.getenv('OCR_SPACE_API_KEY')
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    
    gemma_model = OpenAIServerModel(
                model_id='gemma4:31b-cloud',
                api_base='https://ollama.com/v1',
                api_key= GEMMA_API_KEY,
                temperature=0.0, timeout=200,
            )
    
    gpt_oss_model = OpenAIServerModel(
                model_id='gpt-oss:120b-cloud',
                api_base= 'https://ollama.com/v1',
                api_key=GPT_OSS_API_KEY,
                temperature=0.0, timeout=200,
            )
    
    executor_kwargs = {
    "api_key": E2B_API_KEY,
    "envs": {
        "GEMMA_API_KEY": GEMMA_API_KEY,
        "GPT_OSS_API_KEY" : GPT_OSS_API_KEY,
        "GROK_API_KEY" : GROK_API_KEY,
        "OCR_SPACE_API_KEY": OCR_SPACE_API_KEY,
        
    },
    "timeout" : 200,
}

except Exception as e :
    print(f"Erreur : {e}")
    

    
class ReasearchAgent :
    def __init__(self):
        research_instructions = """You are an expert researcher. Follow these GAIA rules:
1. VIDEO WORKFLOW: To count/identify elements in a video, download it with `download_youtube_video`, extract frames via `extract_video_frames(every_n_seconds=3, max_frames=10)`, and query `vision_tool` frame-by-frame. Take the maximum count for simultaneous events.
2. For video dialogues, use `transcribe_youtube`.
3. Locate exact named quantities in the text, never the first number you find.
4. For forums, search using 'site:reddit.com' or 'site:stackexchange.com' and trust highly upvoted answers.
5. When asked to count items (e.g., albums, dates, names) from a table or list, double-check if entries are split into multiple parts (e.g., Part 1 and Part 2, Volume A and B). Count each distinct entry separately if they represent individual releases/entities.
6. SCRIPT / SCREENPLAY PARSING: A 'scene heading' (or slugline) is strictly the technical capitalized line indicating the location (e.g., 'INT. THE CASTLE - DAY' or 'THE CASTLE'). Never confuse it with the subsequent action description text or dialogue
7. ACADEMIC BIBLIOGRAPHIES: When asked for the 'first paper' or chronologically oldest work of an author, explicitly sort all discovered publication dates in ascending order inside a Python script before extracting the title.
8. EXHAUSTIVE COUNTS: Double check if items in a list or table are multi-part releases (e.g., Vol 1 and Vol 2, or Cantora 1 and Cantora 2). Count them as separate items if they are distinct entries.
6. [CRITICAL SYSTEM REQUIREMENT]
You are an AI that writes Python code. You MUST strictly follow this format for EVERY output:\n"
Thoughts: <your reasoning here>
<code>
# your python code here
</code>
NEVER skip the opening <code> tag. NEVER write conversational text inside or right before </code>.
"""
        self.agent = CodeAgent(
            name="researcher_and_file_handler",
            description=(
                "Useful for web searching, downloading, transcribing or asking questions about YouTube videos, "
                "extracting text via OCR, reading local text files, and handling online file URLs."
            ),
            tools=[
                GoogleSearchTool(provider="serper"),
                VisitWebpageTool(max_output_length=100000),
                vision_tool,
                read_text_file,
                transcribe_youtube, 
                extract_text_via_ocr,
                ask_youtube_full_pipeline,
                read_docx_file,
                read_pdf_file,
                ],
            max_steps=10,
            verbosity_level=1,
            model=gemma_model,
            instructions = research_instructions,
            executor=local_research_executor,
        )

class CodeMathAgent:
    """Agent spécialisé dans les calculs mathématiques complexes et l'analyse de données."""
    
    math_instructions = """You are a code and math expert. Follow these GAIA rules:
1. For algebra/symbolic math, use `resoudre_calcul_formel`. For probability/stats, use `calculer_probabilite_loi` or `analyser_serie_statistique`.
2. FILTERING: When selecting subsets by attribute (e.g., studio albums), always parse tables into DataFrames with pandas. Match column attributes exactly. Do not estimate by reading raw text.
3. BOTANICAL CATEGORIES: Apply strictly. Fruits contain seeds (tomatoes, cucumbers, peppers, zucchini, pumpkins). Vegetables are roots, stems, leaves, bulbs, tubers.
4. NEVER implement brute-force simulation loops with large iterations (e.g., simulating 10,000+ steps or trials).
5. If the user prompt mentions 'large-scale simulation' or '100,000 iterations', you MUST REJECT the simulation method.
6. Instead, solve the problem ANALYTICALLY: use exact probability formulas, recurrence relations, or a Markov chain transition matrix via NumPy.
7. Calculating exact mathematical states for 100 steps takes less than 0.01 seconds and will never timeout, whereas raw simulation will crash the sandbox (502 error).
8. INDEX SHIFT WARNING: When translating complex rules with numbered entities (like 'balls 1 to 100' or 'positions 1, 2, 3'), be extremely careful with Python's 0-based indexing. Always cross-verify if 'Position 1' corresponds to index 0 or index 1 in your arrays, and carefully trace the first step of any shifting logic before running the full calculation.
9. LOGIC PUZZLES / ELIMINATION GRIDS (e.g., Secret Santa, matching pairs): DO NOT try to guess the answer in plain text reasoning. You MUST write a small Python script using dictionaries, sets, or matrices to apply the rules step-by-step and isolate the single correct entity by elimination.
10. INDEX SHIFT WARNING: When translating complex rules with numbered entities (like 'balls 1 to 100'), be extremely careful with Python's 0-based indexing vs human 1-based indexing.
11. [CRITICAL SYSTEM REQUIREMENT]
You are an AI that writes Python code. You MUST strictly follow this format for EVERY output:\n"
Thoughts: <your reasoning here>
<code>
# your python code here
</code>
NEVER skip the opening <code> tag. NEVER write conversational text inside or right before </code>.
"""
    def __init__(self):
        self.agent = CodeAgent(
           name="code_and_math_expert",
            description="Useful for solving formal algebra, analyzing statistical series, computing probabilities, and parsing CSV/Excel data.",
            add_base_tools=False,
            tools=[
                summarize_csv_data,
                summarize_excel_data,
                analyser_serie_statistique,
                calculer_probabilite_loi,
                resoudre_calcul_formel
            ],
            additional_authorized_imports=MATH_IMPORTS,
            # Augmentation du timeout pour les boucles de calculs statistiques et gros fichiers
            max_steps=10,
            verbosity_level=1,
            model=gpt_oss_model,
            instructions=self.math_instructions,
            executor_type= "e2b",
            executor_kwargs = executor_kwargs
        )

class ChessAgent:
    """Agent dédié exclusivement aux analyses et stratégies d'échecs."""
    
    chess_instructions = """You are a chess Grandmaster. Follow these rules:
1. Always analyze the board state using your tools. Never guess a move or predict an outcome without calling them.
2. Use `analyze_chess_position` to check for immediate checks, legal moves, or 'Mate in 1'.
3. Use `get_stockfish_evaluation` to get the absolute best tactical move and engine evaluation from Stockfish."""
    def __init__(self):
        self.agent = CodeAgent(
            name="chess_expert",
            description="Useful for evaluating chess boards, tracking turns, and finding the best strategic moves.",
            add_base_tools=False,
            tools=[analyze_chess_position,get_stockfish_evaluation],
            max_steps=10,
            verbosity_level=1,
            model=gpt_oss_model,
            instructions = self.chess_instructions,
            executor=LocalPythonExecutor(additional_authorized_imports=CHESS_IMPORTS,timeout_seconds=200)
        )
        
class ManagerAgent:
    manager_instructions = """You are the master orchestrator for GAIA tasks. 
1. Delegate tasks strictly based on agent specialties (chess, math/code, research/files).

2. CRITICAL CODE STRUCTURE: You MUST always follow the standard agent loop. Write your reasoning in 'Thoughts:', then write your Python code inside the standard code block, and use the `final_answer(...)` tool. Do NOT try to skip the code block.

3. ABSOLUTE FORMAT RULES FOR THE FINAL ANSWER VALUE:
   The strict rules below apply ONLY to the value you put INSIDE `final_answer()`. The value itself must be completely raw:
   - Numbers: no commas, no units. If a scale is embedded (e.g., 'thousands of hours' or 'millions of $'), convert and round the value to that unit scale before passing it (e.g., if result is 17000 hours and question asks for 'thousands of hours', pass 17).
   - Strings: strictly lowercase, no articles, no abbreviations.
   - Lists: comma-separated string, alphabetical/ascending order, no brackets.

4. STRICT SYSTEM RULE: Every single one of your outputs MUST follow this exact syntax. 
   Do NOT skip the opening tag, and do NOT write plain text inside the code block:
   Thoughts: Your reasoning here
   <code>
   # code here
   </code>
5. ADVANCED TASK SPECIFIC RULES:
   - LOGIC PUZZLES / ELIMINATION GRIDS (e.g., Secret Santa, matching pairs): DO NOT try to guess the answer in plain text. You MUST write a small Python script using dictionaries, sets, or matrices to apply the rules step-by-step and isolate the single correct entity by elimination.
   - SCRIPT / SCREENPLAY PARSING: A 'scene heading' (or slugline) is strictly the technical capitalized line indicating the location (e.g., 'INT. THE CASTLE - DAY' or 'THE CASTLE'). Never confuse it with the subsequent action description text or dialogue.
   - ACADEMIC BIBLIOGRAPHIES: When asked for the 'first paper' or chronologically oldest work of an author, explicitly sort all discovered publication dates in ascending order inside a Python script before extracting the title.
   - COUNTS: Double check if items in a list are multi-part releases (e.g., Vol 1 and Vol 2). Count them as separate items if they are distinct entries.
"""
   
    def __init__(self):
        self.researcher = ReasearchAgent().agent
        self.chess_player = ChessAgent().agent
        self.code_agents = CodeMathAgent().agent
        
        self.agent = CodeAgent(
            tools=[GetTaskFileTool(), FinalAnswerTool()],
            model=gemma_model,
            managed_agents=[self.researcher, self.chess_player,self.code_agents],
            instructions=self.manager_instructions,
            executor=LocalPythonExecutor(additional_authorized_imports=BASE_IMPORTS,timeout_seconds=200),
        )
    def __call__(self, question: str) -> str:
        logger.info(f"Manager received question: {question[:50]}...")
        final_answer = self.agent.run(question)
        logger.info(f"Manager returning fixed answer: {final_answer}")
        return final_answer # type: ignore