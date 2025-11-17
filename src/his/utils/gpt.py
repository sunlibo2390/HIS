import os
from typing import Callable, Iterable, List, Optional

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "128"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_API_BASE") or None,
)


def _run_completion(
    messages: Iterable[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: Optional[str] = None,
    n: int = 1,
) -> List[str]:
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
    )
    return [choice.message.content for choice in response.choices]


def gpt_api(prompt: str, n: int = 1, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS) -> List[str]:
    return _run_completion(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
    )


def gpt_iteration_api(
    prompt_list: List[str],
    temperature: float = 0.0,
    clean: Callable[[str], str] = lambda x: x,
) -> List[str]:
    messages: List[dict] = []
    outputs: List[str] = []
    for prompt in prompt_list:
        messages.append({"role": "user", "content": prompt})
        response = _run_completion(messages, temperature=temperature, max_tokens=1024)[0]
        cleaned = clean(response)
        outputs.append(cleaned)
        messages.append({"role": "assistant", "content": cleaned})
    return outputs


def gpt_dialog_completion(
    dialog: List[dict],
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: Optional[str] = None,
) -> str:
    while True:
        try:
            return _run_completion(
                dialog,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )[0]
        except Exception:
            continue
