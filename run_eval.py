import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
import numpy as np
import pandas as pd
import tenacity
from loguru import logger
from omegaconf import DictConfig, open_dict
from pydantic import BaseModel
from tqdm import tqdm
import torch, gc
import torch, gc



logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logger.opt(colors=True)

DEFAULT_SYSTEM_MESSAGE = "Sei un assistente utile."

QUERY_TEMPLATE_MULTICHOICE = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. L'ultima riga della tua risposta deve essere nel seguente formato: 'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Ragiona brevemente prima di rispondere.

{question}

{options}
""".strip()

QUERY_TEMPLATE_MULTICHOICE_FAST = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.

{question}

{options}

Risposta:
""".strip()


class ProviderEnum(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    CUSTOM_OPENAI = "custom_openai"
    VLLM = "vllm"

class Sample(BaseModel):
    messages: List[Dict[str, str]]
    answer: str


class ChatCompletionRequest(BaseModel):
    index: int
    provider: ProviderEnum
    model: str
    messages: List[Dict[str, str]]
    answer: str
    temperature: float = 0.7
    max_tokens: int = 150
    fast: bool = False


class ChatCompletionResponse(ChatCompletionRequest):
    output: str


class RateLimiter(object):
    """Easy peasy rate limiter to throttle requests."""

    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.request_times = deque()

    def wait(self):
        current_time = time.time()
        with self.lock:
            # remove old ones
            while self.request_times and current_time - self.request_times[0] >= 60:
                self.request_times.popleft()

            if len(self.request_times) < self.rate:
                self.request_times.append(current_time)
                return True
            else:
                return False

    def throttle_requests(self):
        """Wait until it can proceed."""
        while not self.wait():
            time.sleep(0.1)

    def get_total_requests(self):
        """Return the total number of requests processed."""
        with self.lock:
            return len(self.request_times)


class BaseProvider(ABC):
    @abstractmethod
    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        pass


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str):
        import google.generativeai as genai  # type: ignore

        self.genai = genai
        self.genai.configure(api_key=api_key)

    @staticmethod
    def _change_role(message: Dict[str, str]) -> Dict[str, str]:
        return {"role": message["role"], "parts": message["content"]}

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }
        mq = self.genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=DEFAULT_SYSTEM_MESSAGE,
        )

        history = [self.change_role(message) for message in messages[1:-1]]
        chat_session = mq.start_chat(history=history)

        response = chat_session.send_message(messages[-1]["content"])

        return response.text.strip()


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key, **kwargs)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        from anthropic import Anthropic  # type: ignore

        self.client = Anthropic(api_key=api_key)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        response = self.client.messages.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=messages[0]["content"],
            messages=messages[1:],
        )

        return response.content[0].text.strip()


def model_factory(provider: ProviderEnum, api_key: str, **kwargs) -> BaseProvider:
    match provider:
        case ProviderEnum.OPENAI:
            return OpenAIProvider(api_key=api_key)
        case ProviderEnum.ANTHROPIC:
            return AnthropicProvider(api_key=api_key)
        case ProviderEnum.GOOGLE:
            return GoogleProvider(api_key=api_key)
        case ProviderEnum.CUSTOM_OPENAI:
            return OpenAIProvider(api_key=api_key, **kwargs)
        case ProviderEnum.VLLM:
            model = kwargs.pop("model")
            return VLLMProvider(model=model, **kwargs)
        case _:
            raise ValueError(f"Provider {provider} not supported.")

class Provider(BaseProvider):
    def __init__(self, api_key: str, provider: ProviderEnum, **kwargs):
        self.provider = model_factory(provider, api_key, **kwargs)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        return self.provider.complete(model, messages, temperature, max_tokens)


def configure_payload(
    topic: str,
    question: str,
    options: List[Dict[str, str]],
    answer: str,
    fast: bool = False,
    system_message: str = None,
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> Sample:
    def format_options(options: List[Dict[str, str]]) -> str:
        formatted_options = "\n".join(
            [f"{list(item.keys())[0]}) {list(item.values())[0]}" for item in options]
        )
        keys = "".join([list(item.keys())[0] for item in options])

        return formatted_options, keys

    USER_QUERY_TEMPLATE = (
        QUERY_TEMPLATE_MULTICHOICE_FAST if fast else QUERY_TEMPLATE_MULTICHOICE
    )

    options_str, merged_letters = format_options(options)
    messages = [
        {
            "role": "system",
            "content": system_message or DEFAULT_SYSTEM_MESSAGE,
        },
    ]

    if few_shots:
        for shot in few_shots:
            shot_options, shot_merged_letters = format_options(shot["options"])
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": USER_QUERY_TEMPLATE.format(
                            topic=shot["category"],
                            question=shot["question"],
                            options=shot_options,
                            merged_letters=shot_merged_letters,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": shot["answer"],
                    },
                ]
            )

    messages.append(
        {
            "role": "user",
            "content": USER_QUERY_TEMPLATE.format(
                topic=topic,
                question=question,
                options=options_str,
                merged_letters=merged_letters,
            ),
        }
    )

    return Sample(messages=messages, answer=answer)


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(Exception),
    wait=tenacity.wait_exponential(multiplier=1, max=5),
    stop=tenacity.stop_after_attempt(50),
)
def process_request(
    request: ChatCompletionRequest,
    client: BaseProvider,
    rate_limiter: Optional[RateLimiter] = None,
) -> ChatCompletionResponse:
    if rate_limiter:
        rate_limiter.throttle_requests()
    completion = client.complete(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    return ChatCompletionResponse(**request.dict(), output=completion)


def load_few_shots(file_path: str) -> List[Dict[str, Any]]:
    few_shots = pd.read_json(file_path, lines=True)
    return few_shots.to_dict(orient="records")


def extract_answer_fast(output: str) -> str:
    # First try: ASSISTANT: X
    match = re.search(r"ASSISTANT:\s*([A-Z])", output)
    if match:
        return match.group(1)

    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    min_index = min(
        [output.find(letter) for letter in LETTERS if letter in output], default=-1
    )
    if min_index == -1:
        return ""
    return output[min_index]


def extract_answer(output: str) -> str:
    # First try: ASSISTANT: X
    match = re.search(r"ASSISTANT:\s*([A-Z])", output)
    if match:
        return match.group(1)

    def _find(pattern: str, text: str, ignore_case: bool = True) -> str:
        flags = re.DOTALL | (re.IGNORECASE if ignore_case else 0)
        match = re.search(pattern, text, flags)
        if match:
            answer = re.sub(r"[è:)(?+-,;.]", "", match.group(1)).strip()
            answer = re.sub(r"^(?:sarà\s+la\s+|la\s+)?", "", answer).strip()
            return extract_answer_fast(answer) if answer else ""
        return ""

    def_pattern = r"Risposta:\s*(.*?)\s*(?=\n[A-Z]\)|\Z)"

    fallback_patterns = [
        r"quindi, la risposta è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"risposta\s*(?:corretta|giusta|appropriata|esatta|migliore|ottimale|finale|definitiva)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"risposta\s*più\s*(?:corretta|appropriata)\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:soluzione|opzione|scelta|alternativa)\s*(?:corretta)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:quindi|in\s*conclusione,?)?\s*(?:la\s*)?risposta\s*è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:la\s*)?(?:risposta|opzione|scelta)\s*(?:corretta|giusta|esatta)\s*è\s*(?:la\s*)?(?:lettera\s*)?([A-Z])",
    ]

    answer = _find(def_pattern, output, ignore_case=False)
    if answer:
        return answer

    if "nessuna delle opzioni" in output.lower():
        return ""

    for pattern in fallback_patterns:
        answer = _find(pattern, output)
        if answer:
            return answer

    return ""


def _compute_stat(values: list, stat: str):
    stat_functions = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }
    if stat not in stat_functions:
        raise ValueError(f"Unknown {stat =}")
    return stat_functions[stat](values)


def aggregate_results(
    responses: List[ChatCompletionResponse],
    default_stats: Tuple[str, ...] = ("mean", "std"),
    name2stats: Dict[str, Tuple[str]] | None = None,
) -> Dict[str, float]:
    """
    Aggregate results from multiple evaluations into a single result dictionary.
    Similar to OpenAI simple_eval.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)

    for resp in responses:
        predicted_answer = (
            extract_answer_fast(resp.output)
            if not resp.fast
            else extract_answer_fast(resp.output)
        )
        correct = predicted_answer == resp.answer
        name2values["accuracy"].append(float(correct))

    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)

    return final_metrics


def save_intermediate_results(
    responses: List[ChatCompletionResponse], output_file: Path
):
    results = [resp.dict() for resp in sorted(responses, key=lambda x: x.index)]

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def load_intermediate_results(
    output_file: Path,
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    if not output_file.exists():
        logger.info(f"Checkpoint file not found, starting from scratch.")
        return [], set()

    logger.info(f"Loading checkpoint file {output_file}")
    with open(output_file, "r") as f:
        results = json.load(f)
        results = [ChatCompletionResponse(**r) for r in results]

    ids = set([r.index for r in results])
    return results, ids


def process(
    requests: List[ChatCompletionRequest], client: Provider, config: DictConfig
):

    if config.limit:
        requests = requests[: config.limit]

    Path(config.data.output_dir).mkdir(parents=True, exist_ok=True)

    dataset_name = Path(config.data.data_file).stem

    clean_model = (
        lambda x: x.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
    )

    output_file = (
        Path(config.data.output_dir)
        / f"{config.provider}_{clean_model(config.model)}_{dataset_name}.json"
    )

    logger.info(
        f"\n\nmodel: {config.model}\nthreads: {config.num_threads}\nfile: {config.data.data_file}\n"
    )

    all_responses = []

    # -------------------------
    # FAST PATH FOR VLLM
    # -------------------------
    if config.provider == "vllm":

        logger.info("Using vLLM batched inference")

        outputs = client.provider.batch_complete(
            requests,
            config.temperature,
            config.max_tokens,
        )

        for req, text in zip(requests, outputs):
            resp = ChatCompletionResponse(**req.dict(), output=text)
            all_responses.append(resp)

    # -------------------------
    # NORMAL PATH (API MODELS)
    # -------------------------
    else:

        rate_limiter = None
        if config.rate_limiting.enabled:
            rate_limiter = RateLimiter(config.rate_limiting.requests_per_minute)

        pbar = tqdm(total=len(requests), desc="Processing responses")

        with ThreadPoolExecutor(
            max_workers=min(config.num_threads, len(requests))
        ) as executor:

            futures = [
                executor.submit(process_request, req, client, rate_limiter)
                for req in requests
            ]

            for future in as_completed(futures):
                resp = future.result()
                all_responses.append(resp)
                pbar.update(1)

    metrics = aggregate_results(all_responses)

    logger.info(f"Metrics: {metrics}")

    results = [resp.dict() for resp in sorted(all_responses, key=lambda x: x.index)]

    with open(output_file, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def load_requests(config: DictConfig) -> List[ChatCompletionRequest]:
    df = pd.read_json(config.data.data_file, lines=True)
    data = df.to_dict(orient="records")

    few_shots = (
        load_few_shots(config.data.few_shot_file) if config.data.few_shot_file else None
    )
    requests = []
    for i, item in tqdm(enumerate(data), total=len(data), desc="Preparing requests"):
        sample = configure_payload(
            topic=item["category"],
            question=item["question"],
            options=item["options"],
            answer=item["answer"],
            fast=config.fast,
            system_message=config.system_message,
            few_shots=few_shots,
        )
        requests.append(
            ChatCompletionRequest(
                index=i,
                provider=ProviderEnum(config.provider),
                model=config.model,
                messages=sample.messages,
                answer=sample.answer,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                fast=config.fast,
            )
        )
    return requests

class VLLMProvider(BaseProvider):
    def __init__(self, model: str, **kwargs):
        from vllm import LLM

        self.model_name = model
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            **kwargs,
        )

    # This is required by the abstract base class
    def complete(
        self,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Wrap single request in batch_complete
        from types import SimpleNamespace
        dummy_req = SimpleNamespace(messages=messages)
        # batch_complete expects a list of ChatCompletionRequest objects
        # we can wrap a minimal object for compatibility
        class DummyRequest:
            def __init__(self, messages):
                self.messages = messages
        out = self.batch_complete([DummyRequest(messages)], temperature, max_tokens)
        return out[0]

    def batch_complete(
        self,
        requests: list,
        temperature: float,
        max_tokens: int,
    ) -> list:

        from vllm import SamplingParams

        prompts = []
        for req in requests:
            prompt = ""
            for m in req.messages:
                prompt += f"{m['role'].upper()}: {m['content']}\n"
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        return [o.outputs[0].text.strip() for o in outputs]

@hydra.main(version_base=None, config_path="./", config_name="config")
def run(config: DictConfig):

    for model_cfg in config.models:
        provider_str = model_cfg["provider"]
        model_name = model_cfg["model"]
        api_key = model_cfg.get("api_key", "")

        provider_kwargs = dict(model_cfg.get("provider_kwargs", {}))
        if provider_str == "vllm":
            provider_kwargs["model"] = model_name

        with open_dict(config):
            config.provider = provider_str
            config.model = model_name
            config.api_key = api_key

        logger.info(f"<bold>🔎 Running evaluation: {provider_str}:{model_name} 🔎</bold>")

        client = Provider(
            api_key=config.api_key,
            provider=ProviderEnum(config.provider),
            **provider_kwargs,
        )

        requests = load_requests(config)
        process(requests=requests, client=client, config=config)

        # ---- free GPU memory ----
        if provider_str == "vllm":
            del client.provider.llm
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Cleared GPU memory for model {model_name}")

if __name__ == "__main__":
    run()
