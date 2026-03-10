from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_env_file(env_path: Union[str, Path] = ".env") -> None:
    """Load KEY=VALUE pairs from .env if present (without overriding existing env vars)."""
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class _ForwardCache:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    next_token_logits: torch.Tensor


class Model:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self._cache: Optional[_ForwardCache] = None
        self.inference_batch_size = 8

    def _run_model_dynamic_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Run model forward with adaptive micro-batching to reduce OOM risk."""
        generation_config = generation_config or {}

        total_batch = input_ids.size(0)
        logits_chunks: List[torch.Tensor] = []
        cursor = 0
        current_batch_size = max(1, int(self.inference_batch_size))

        while cursor < total_batch:
            chunk_size = min(current_batch_size, total_batch - cursor)
            chunk_input_ids = input_ids[cursor:cursor + chunk_size]
            chunk_attention_mask = attention_mask[cursor:cursor + chunk_size]

            try:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        **generation_config,
                    )
                logits_chunks.append(outputs.logits)
                cursor += chunk_size
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)

        self.inference_batch_size = current_batch_size
        return torch.cat(logits_chunks, dim=0)

    def forward(
        self,
        text: Union[str, Sequence[str]],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        generation_config = generation_config or {}
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        logits = self._run_model_dynamic_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        next_token_logits = logits[:, -1, :]
        self._cache = _ForwardCache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            next_token_logits=next_token_logits,
        )
        return next_token_logits

    __call__ = forward

    def _sequence_log_prob_and_counts(
        self,
        text: Union[str, Sequence[str]],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        generation_config = generation_config or {}
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        logits = self._run_model_dynamic_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        next_tokens = input_ids[:, 1:]
        token_log_probs = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)

        valid_mask = attention_mask[:, 1:].to(token_log_probs.dtype)
        sequence_log_probs = (token_log_probs * valid_mask).sum(dim=1)
        token_counts = valid_mask.sum(dim=1).clamp_min(1.0)

        return sequence_log_probs, token_counts

    def log_prob(
        self,
        text: Union[str, Sequence[str]],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        sequence_log_probs, _ = self._sequence_log_prob_and_counts(
            text=text,
            generation_config=generation_config,
        )
        return sequence_log_probs

    def familiarity_score(
        self,
        text: Union[str, Sequence[str]],
        generation_config: Optional[Dict[str, Any]] = None,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Length-normalized familiarity score.

        score = log p(x) / (token_count ** length_penalty)
        """
        if length_penalty < 0:
            raise ValueError("length_penalty는 0 이상의 값이어야 합니다.")

        sequence_log_probs, token_counts = self._sequence_log_prob_and_counts(
            text=text,
            generation_config=generation_config,
        )
        return sequence_log_probs / torch.pow(token_counts, length_penalty)

    def _get_eos_token_ids(self) -> set[int]:
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return set()
        if isinstance(eos_token_id, int):
            return {eos_token_id}
        return set(eos_token_id)

    def sample(
        self,
        tokens: int = 1,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if self._cache is None:
            raise ValueError("forward 또는 __call__을 먼저 실행해야 sample을 사용할 수 있습니다.")
        if tokens < 1:
            raise ValueError("tokens는 1 이상의 정수여야 합니다.")

        generation_config = generation_config or {}
        eos_token_ids = self._get_eos_token_ids()

        input_ids = self._cache.input_ids
        attention_mask = self._cache.attention_mask
        current_logits = self._cache.next_token_logits

        batch_size = input_ids.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        generated_token_ids: List[List[int]] = [[] for _ in range(batch_size)]

        for _ in range(tokens):
            probs = torch.softmax(current_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            for i in range(batch_size):
                if finished[i]:
                    next_token[i, 0] = self.tokenizer.pad_token_id
                    continue
                token_id = int(next_token[i, 0].item())
                generated_token_ids[i].append(token_id)
                if token_id in eos_token_ids:
                    finished[i] = True

            input_ids = torch.cat([input_ids, next_token], dim=1)
            next_mask = torch.ones(
                (attention_mask.size(0), 1),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([attention_mask, next_mask], dim=1)

            if bool(finished.all()):
                break

            logits = self._run_model_dynamic_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            current_logits = logits[:, -1, :]

        self._cache = _ForwardCache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            next_token_logits=current_logits,
        )

        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_token_ids
        ]

    def ensembled_sample(
        self,
        texts: Sequence[str],
        tokens: int = 1,
        ensemble_scheme: str = "sum",
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not texts:
            raise ValueError("texts는 비어 있을 수 없습니다.")
        if tokens < 1:
            raise ValueError("tokens는 1 이상의 정수여야 합니다.")

        generation_config = generation_config or {}
        eos_token_ids = self._get_eos_token_ids()

        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        sampled: List[int] = []

        for _ in range(tokens):
            logits = self._run_model_dynamic_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

            last_token_index = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=self.device)
            batch_logits = logits[batch_indices, last_token_index, :]

            if ensemble_scheme == "sum":
                ensembled_logits = batch_logits.sum(dim=0, keepdim=True)
            elif ensemble_scheme == "mean":
                ensembled_logits = batch_logits.mean(dim=0, keepdim=True)
            elif ensemble_scheme == "max":
                ensembled_logits, _ = batch_logits.max(dim=0, keepdim=True)
            else:
                raise ValueError("지원하지 않는 ensemble_scheme입니다. (sum/mean/max)")

            probs = torch.softmax(ensembled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            sampled.append(token_id)

            repeated = next_token.repeat(input_ids.size(0), 1)
            input_ids = torch.cat([input_ids, repeated], dim=1)
            next_mask = torch.ones(
                (attention_mask.size(0), 1),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([attention_mask, next_mask], dim=1)

            if token_id in eos_token_ids:
                break

        return self.tokenizer.decode(sampled, skip_special_tokens=True)


class ModelAPI:
    def __init__(
        self,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: int = 60,
        env_path: Union[str, Path] = ".env",
    ) -> None:
        _load_env_file(env_path)

        self.url = url or os.getenv("MODEL_API_URL")
        self.model_name = model_name or os.getenv("MODEL_API_MODEL_NAME")
        self.auth_token = auth_token or os.getenv("MODEL_API_TOKEN")
        self.timeout = timeout

        if not self.url:
            raise ValueError("ModelAPI url이 필요합니다. 인자 또는 .env(MODEL_API_URL)로 설정하세요.")
        if not self.model_name:
            raise ValueError(
                "ModelAPI model_name이 필요합니다. 인자 또는 .env(MODEL_API_MODEL_NAME)로 설정하세요."
            )

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def forward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        generation_config = generation_config or {}
        if messages is None:
            if prompt is None:
                raise ValueError("prompt 또는 messages 중 하나는 반드시 제공되어야 합니다.")
            messages = [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            **generation_config,
        }
        response = requests.post(
            self.url,
            json=payload,
            headers=self._build_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    __call__ = forward


class model(Model):
    pass


class model_api(ModelAPI):
    pass
