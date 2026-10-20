import re
from typing import Any

from config import build_agent
from utils.logger import get_logger

logger = get_logger(__name__)

_reasoning_model = build_agent('gpt-oss:120b-cloud')
_formatting_model = build_agent('gpt-oss:120b-cloud')


def _extract_verdict(output: str) -> bool:
    """Retourne True si le verdict final est PASS."""
    if not output or not output.strip():
        raise Exception("Evaluator model returned empty response.")

    # Approche 1 : la dernière ligne contient exactement PASS ou FAIL
    lines = output.strip().splitlines()
    last_line = lines[-1].strip().upper()
    if last_line in ("PASS", "FAIL"):
        return last_line == "PASS"

    # Approche 2 : chercher la dernière occurrence de PASS/FAIL en tant que mot
    matches = list(re.finditer(r'\b(PASS|FAIL)\b', output))
    if matches:
        return matches[-1].group(1) == "PASS"

    raise Exception(f"No clear PASS/FAIL verdict found in output:\n{output}")


def _run_judge(model: Any, prompt: str, label: str) -> bool:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    try:
        output = str(model(messages).content)
    except Exception as e:
        logger.error(f"[{label}] Model call failed: {e}")
        raise

    logger.info(f"[{label}] Feedback:\n{output}")

    try:
        return _extract_verdict(output)
    except Exception as e:
        # Lève une exception avec le texte complet pour aider l'agent à comprendre l'échec
        raise Exception(f"[{label}] {e}\nFull evaluator output:\n{output}")


def check_reasoning(final_answer, agent_memory, agent=None) -> bool:
    prompt = f"""
Task & steps: {agent_memory.get_succinct_steps()}
Answer: {final_answer}

Evaluate if the final answer is factually correct AND properly derived from the source material.
For video questions, the answer must be based on direct analysis of the video content (frames/transcript), not on indirect web search results that may not correspond to the specific video.
If the agent merely searched the web and picked a number without video evidence, consider it a FAIL.
List reasons for PASS/FAIL, then final verdict: PASS or FAIL.
"""
    return _run_judge(_reasoning_model, prompt, "check_reasoning")


def ensure_formatting(final_answer, agent_memory, agent=None) -> bool:
    prompt = f"""
Task & steps: {agent_memory.get_succinct_steps()}
FINAL ANSWER: {final_answer}

Evaluate format compliance:
- Answer type: number OR minimal words OR comma-separated list (no brackets).
- Number: no commas, no unit/% unless specified, Arabic digits (e.g., 9,3,1093).
- Currency: use symbol ($40.00) if asked.
- String: lowercase, no articles, no abbreviations, digits in words unless specified.
- List: elements follow number/string rules; no brackets; order as requested (alpha/asc).
- CRITICAL: If the task embeds a scale ("thousand X", "millions Y", "percentage"), the answer must already be in that unit (e.g., 17000h → 17 for "thousand hours"). If not, FAIL.
- If list order is specified (alphabetical/ascending), verify it matches. If not, FAIL.

List reasons, then final verdict on the LAST LINE: PASS or FAIL.
"""
    return _run_judge(_formatting_model, prompt, "ensure_formatting")