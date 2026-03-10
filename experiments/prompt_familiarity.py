from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_judge_messages(judge_prompt_path: str) -> List[Dict[str, str]]:
    path = Path(judge_prompt_path)
    spec = importlib.util.spec_from_file_location("judge_prompt_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"judge prompt 파일을 불러올 수 없습니다: {judge_prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "LLM_JUDGE_MESSAGES"):
        raise ValueError("judge prompt 파일에 LLM_JUDGE_MESSAGES가 없습니다.")

    return getattr(module, "LLM_JUDGE_MESSAGES")


def extract_judge_score(response: Dict[str, Any]) -> tuple[float, str]:
    if "score" in response:
        return float(response["score"]), str(response.get("reason", ""))

    text = ""
    if "choices" in response and response["choices"]:
        text = response["choices"][0].get("message", {}).get("content", "")

    if not text:
        raise ValueError("judge 응답에서 score를 찾지 못했습니다.")

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    payload = json.loads(match.group(0) if match else text)
    return float(payload["score"]), str(payload.get("reason", ""))


def rankdata(values: List[float]) -> List[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j < len(sorted_pairs) and sorted_pairs[j][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt familiarity vs performance experiment")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--judge_prompt_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    args = parser.parse_args()

    from src.models import Model, ModelAPI
    from src.prompt import Prompt
    from src.utils.output import make_output_dir, save_json

    data = load_jsonl(args.data_path)
    questions = [x["question"] for x in data]

    model = Model(args.model_path)
    model.inference_batch_size = args.generation_batch_size

    familiarities: List[float] = []
    outputs: List[str] = []

    for start in range(0, len(questions), args.generation_batch_size):
        batch_questions = questions[start:start + args.generation_batch_size]
        batch_familiarity = model.familiarity_score(
            batch_questions,
            length_penalty=args.length_penalty,
        )
        familiarities.extend(batch_familiarity.detach().cpu().tolist())

        model.forward(batch_questions)
        batch_outputs = model.sample(tokens=args.max_new_tokens)
        outputs.extend(batch_outputs)

    judge_messages_template = load_judge_messages(args.judge_prompt_path)
    prompt_builder = Prompt(judge_messages_template)
    judge = ModelAPI()

    records: List[Dict[str, Any]] = []
    judge_scores: List[float] = []

    for sample, familiarity, output in zip(data, familiarities, outputs):
        rendered_messages = prompt_builder.to_chat_completion(
            {
                "question": sample["question"],
                "answer": sample["answer"],
                "output": output,
            }
        )
        judge_response = judge.forward(messages=rendered_messages)
        score, reason = extract_judge_score(judge_response)
        judge_scores.append(score)

        records.append(
            {
                "question": sample["question"],
                "answer": sample["answer"],
                "output": output,
                "familiarity_score": familiarity,
                "judge_score": score,
                "judge_reason": reason,
                "raw_judge_response": judge_response,
            }
        )

    pearson = pearson_corr(familiarities, judge_scores)
    spearman = pearson_corr(rankdata(familiarities), rankdata(judge_scores))

    output_dir = make_output_dir()

    scatter_path = output_dir / "familiarity_vs_performance_scatter.png"
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(familiarities, judge_scores, alpha=0.8)
        plt.xlabel("Familiarity Score")
        plt.ylabel("Judge Score")
        plt.title("Familiarity vs Judge Score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=150)
        plt.close()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib이 필요합니다. `pip install matplotlib` 후 다시 실행하세요."
        ) from exc

    save_json(records, output_dir / "sample_results.json")
    save_json(
        {
            "pearson": pearson,
            "spearman": spearman,
            "num_samples": len(records),
            "scatter_plot": str(scatter_path),
        },
        output_dir / "correlation.json",
    )

    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
