"""
Harnais d'evaluation GAIA (Phase 5 du plan).

Charge le split de validation, filtre par niveau, fait tourner le manager
sur chaque question, et calcule le score exact-match (methode GAIA officielle).

Prerequis :
    - Avoir accepte les conditions du dataset sur
      https://huggingface.co/datasets/gaia-benchmark/GAIA
    - Etre connecte : `huggingface-cli login` (ou variable d'env HF_TOKEN)
    - pip install datasets

Usage :
    python run_eval.py --level 1 --limit 10
"""
import logging

import argparse
import csv
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

# from agent import build_agents
from gaia_scorer import question_scorer
from agent import ManagerAgent


# Crée le dossier workspace s'il n'existe pasogging.basicConfig(level=logging.INFO)

def load_validation_questions(level: int):
    """Telecharge le dataset GAIA en local (necessaire pour resoudre les fichiers
    joints via file_path) et charge le split de validation pour le niveau demande."""
    data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
    config = f"2023_level{level}"
    dataset = load_dataset(data_dir, config, split="validation")
    return dataset, data_dir


def run_evaluation(level: int, limit: int | None, output_csv: str):
    questions, data_dir = load_validation_questions(level)
    if limit:
        questions = questions.select(range(min(limit, len(questions))))

    manager = ManagerAgent()

    results = []
    correct = 0

    for i, row in enumerate(questions): # type: ignore
        row: dict = row  # type: ignore[no-redef]
        question = row["Question"]
        ground_truth = row["Final answer"]
        task_id = row["task_id"]

        # Certaines questions ont une piece jointe : on resout son chemin local
        # et on l'ajoute au texte envoye au manager pour que le file_agent puisse la lire.
        file_path = row.get("file_path", "")
        if file_path:
            full_path = os.path.join(data_dir, file_path)
            full_question = f"{question}\n\n(Fichier joint disponible ici : {full_path})"
        else:
            full_question = question

        print(f"\n[{i + 1}/{len(questions)}] (niveau {row['Level']}) {question[:80]}...")

        try:
            prediction = manager(full_question)
        except Exception as e:
            prediction = f"__ERREUR__: {e}"

        is_correct = question_scorer(str(prediction), str(ground_truth))
        correct += int(is_correct)

        print(f"  Attendu : {ground_truth}")
        print(f"  Obtenu  : {prediction}")
        print(f"  {'CORRECT' if is_correct else 'INCORRECT'}")

        results.append({
            "task_id": task_id,
            "level": row["Level"],
            "question": question,
            "file_path": file_path,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": is_correct,
        })

    total = len(questions)
    accuracy = correct / total if total else 0
    print(f"\n=== Score global : {correct}/{total} ({accuracy:.1%}) ===")

    if results:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Resultats detailles sauvegardes dans : {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalue l'agent sur le split de validation GAIA.")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1,
                         help="Niveau de difficulte a evaluer (defaut : 1)")
    parser.add_argument("--limit", type=int, default=10,
                         help="Nombre max de questions a tester (defaut : 10)")
    parser.add_argument("--output", type=str, default="eval_results.csv",
                         help="Fichier CSV de sortie")
    args = parser.parse_args()

    run_evaluation(args.level, args.limit, args.output)