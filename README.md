# GAIA Agent

GAIA Agent is a sophisticated multi-agent system built with `smolagents`, designed to tackle complex, real-world tasks that require a combination of web research, data analysis, mathematical reasoning, and specialized domain knowledge. It is specifically optimized for the GAIA (General AI Assistants) benchmark.

## 🚀 Architecture

The system employs a hierarchical orchestration pattern where a **Manager Agent** delegates tasks to three specialized sub-agents based on the nature of the request:

### 1. Manager Agent (The Orchestrator)

- **Role**: Acts as the primary entry point and decision-maker.
- **Responsibility**: Analyzes the user's query, breaks it down into sub-tasks, and routes them to the appropriate specialized agent. It ensures the final answer follows strict formatting rules.

### 2. Research Agent

- **Role**: Expert in information retrieval and multimodal content analysis.
- **Capabilities**:
  - **Web Search**: Integrated with Google Search (via Serper).
  - **YouTube Pipeline**: Can transcribe audio (via Groq/Whisper) or analyze video frames using Vision-Language Models (VLM).
  - **Document Parsing**: Reads and extracts text from PDF and DOCX files.
  - **OCR**: Extracts text from images using the OCR.space API.
  - **File Handling**: Downloads files from URLs for further processing.

### 3. Code & Math Agent

- **Role**: Specialist in quantitative analysis and symbolic mathematics.
- **Capabilities**:
  - **Data Analysis**: Summarizes and filters CSV and Excel files using `pandas`.
  - **Symbolic Math**: Solves equations, performs differentiation and integration via `sympy`.
  - **Statistics**: Computes probabilities for Normal and Binomial distributions and analyzes statistical series using `scipy` and `numpy`.
  - **Execution**: Runs code in a secure E2B sandbox.

### 4. Chess Agent

- **Role**: Chess Grandmaster AI.
- **Capabilities**:
  - **Position Analysis**: Analyzes FEN strings to detect checks, mates, and legal moves.
  - **Engine Evaluation**: Integrates with Stockfish 18 to provide the best tactical moves and precise evaluations.

## 🛠️ Features

- **Multimodal Intelligence**: Seamlessly switches between text, image, and video analysis.
- **Secure Code Execution**: Utilizes both local and remote (E2B) Python executors for safe and scalable computation.
- **Domain Specificity**: Dedicated tools for chess and advanced mathematics ensure high precision in specialized tasks.
- **Robust Pipeline**: Specialized workflows for academic bibliographies, script parsing, and exhaustive counting.

## 📦 Installation

### Prerequisites

- Python 3.10+
- An Ollama server (or compatible OpenAI-server) for the LLMs.

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd GAIA_AGENT
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Copy the example environment file and fill in your API keys:
   ```bash
   cp .env.exemple .env
   ```

## 🔑 Environment Variables

The following API keys are required in your `.env` file:

| Variable            | Description                                            |
| :------------------ | :----------------------------------------------------- |
| `GEMMA_API_KEY`     | API key for the Gemma model (via Ollama/OpenAI server) |
| `GPT_OSS_API_KEY`   | API key for the GPT-OSS model                          |
| `E2B_API_KEY`       | API key for the E2B secure code sandbox                |
| `GROK_API_KEY`      | API key for Groq (used for Whisper transcription)      |
| `OCR_SPACE_API_KEY` | API key for OCR.space image-to-text                    |
| `SERPER_API_KEY`    | API key for Serper.dev Google Search                   |

## 💻 Usage

To interact with the agent, you can use the `ManagerAgent` class in `agent.py`:

```python
from agent import ManagerAgent

manager = ManagerAgent()
response = manager("Your complex GAIA task here")
print(response)
```

## 📁 Project Structure

```text
GAIA_AGENT/
├── agent.py              # Main agent definitions and orchestration logic
├── requirements.txt      # Project dependencies
├── .env.exemple          # Template for environment variables
├── tools/                # Custom tool implementations
│   ├── tools.py          # Core tool definitions (Vision, Math, Chess, etc.)
│   ├── get_answer.py      # Task-specific retrieval tools
│   └── ...
└── utils/                # Utility functions (logging, etc.)
```
