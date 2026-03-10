"""LLM-as-a-Judge prompt templates.

Placeholder keys must match experiment fields:
- question
- answer
- output
"""

LLM_JUDGE_PROMPT = """당신은 엄격하고 일관된 한국어 평가자입니다.
아래 Question에 대해 Model Output을 평가하세요.

[평가 기준]
1) 정확성(사실/요구 충족)
2) 관련성(질문 의도 부합)
3) 명확성(이해 용이성)
4) 완성도(누락/불필요 내용 여부)

[출력 형식]
반드시 아래 JSON 형식으로만 답하세요.
{{
  "score": <0부터 10 사이 실수>,
  "reason": "한두 문장 근거"
}}

[Question]
{question}

[Reference Answer]
{answer}

[Model Output]
{output}
"""


LLM_JUDGE_MESSAGES = [
    {
        "role": "system",
        "content": "당신은 LLM 답변 품질을 채점하는 심사위원입니다.",
    },
    {
        "role": "user",
        "content": LLM_JUDGE_PROMPT,
    },
]
