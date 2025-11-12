"""High-level chat orchestration for the RAG demo."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import PromptConfig, RetrievalConfig
from .console import console
from .llm import ModelClientError, ModelInfo, ModelNotFoundError, SupportsModels
from .memory import ConversationMemory
from .retrieval import ContextBuilder


@dataclass
class ChatRequest:
    query: str
    context: str
    prompts: PromptConfig

    def to_messages(self) -> List[dict]:
        context_block = self.prompts.context_template.format(context=self.context)
        return [
            {"role": "system", "content": self.prompts.system_prompt},
            {"role": "user", "content": context_block},
            {
                "role": "user",
                "content": self.prompts.question_template.format(query=self.query),
            },
        ]


class ChatSession:
    def __init__(
        self,
        client: SupportsModels,
        context_builder: ContextBuilder,
        retrieval: RetrievalConfig,
        prompts: PromptConfig,
        provider_name: str,
        memory: ConversationMemory | None = None,
        console_override=None,
    ) -> None:
        self.client = client
        self.context_builder = context_builder
        self.retrieval = retrieval
        self.prompts = prompts
        self.provider_name = provider_name
        self.console = console_override or console
        self.memory = memory

    def list_models(self) -> List[ModelInfo]:
        return self.client.list_models()

    def ensure_model(self, model_name: str | None = None) -> None:
        models = self.list_models()
        if not models:
            raise ModelNotFoundError(
                f"No models were returned by the {self.provider_name} provider."
            )
        if model_name:
            self.client.ensure_model(model_name, models)
            return
        self.client.model = self._prompt_for_model(models)

    def _prompt_for_model(self, models: Sequence[ModelInfo]) -> str:
        table = Table(
            title=f"Available {self.provider_name.title()} Models",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right")
        table.add_column("Model Name", style="magenta")
        for idx, model in enumerate(models, start=1):
            table.add_row(str(idx), model.name)
        self.console.print(table)

        if not sys.stdin.isatty():
            selection = models[0].name
            self.console.print(
                Panel(
                    f"Using {self.provider_name} model '[bold]{selection}[/]'.",
                    title="Model Selection",
                    style="info",
                )
            )
            return selection

        while True:
            user_input = Prompt.ask(
                "Select a model (number or name)",
                default=models[0].name,
            )
            user_input = user_input.strip()
            if not user_input:
                return models[0].name
            if user_input.isdigit():
                idx = int(user_input)
                if 1 <= idx <= len(models):
                    return models[idx - 1].name
                self.console.print("[warning]Invalid selection. Please choose a valid number.[/warning]")
                continue
            for model in models:
                if model.name == user_input:
                    return model.name
            self.console.print("[warning]Model not recognized. Please choose again.[/warning]")

    def build_context(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.context_builder.build_context(
            query,
            top_k=top_k,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )

    def ask(self, query: str) -> str:
        prepared = self._prepare_prompt(query)
        if prepared["from_cache"]:
            cached_answer = prepared["cached_answer"] or ""
            if self.memory:
                self.memory.remember(query, cached_answer, from_cache=True)
            return cached_answer

        answer = self.client.chat(prepared["messages"])
        if self.memory:
            self.memory.remember(query, answer)
        return answer

    def prepare_messages(self, query: str) -> dict:
        """Return the compiled message payload and metadata for *query*."""

        return self._prepare_prompt(query)

    def _prepare_prompt(self, query: str) -> dict:
        memory_context = ""
        memory_tokens = 0
        cached_answer: Optional[str] = None
        if self.memory:
            cached_answer = self.memory.maybe_answer(query)
            if cached_answer is not None:
                return {
                    "messages": [],
                    "context": "",
                    "from_cache": True,
                    "cached_answer": cached_answer,
                    "memory_context": "",
                }
            memory_context, memory_tokens = self.memory.build_memory_context(query)

        max_chars = self.retrieval.max_context_chars
        if max_chars is not None and memory_context:
            max_chars = max(0, max_chars - len(memory_context))
        max_tokens = self.retrieval.max_context_tokens
        if max_tokens is not None and memory_tokens:
            max_tokens = max(0, max_tokens - memory_tokens)

        allow_tokens = max_tokens is None or max_tokens > 0
        allow_chars = max_chars is None or max_chars > 0
        if allow_tokens or allow_chars:
            retrieval_context = self.context_builder.build_context(
                query,
                max_chars=max_chars if allow_chars else None,
                max_tokens=max_tokens if allow_tokens else None,
            )
        else:
            retrieval_context = ""
        context = self._merge_context(memory_context, retrieval_context)
        messages = ChatRequest(query=query, context=context, prompts=self.prompts).to_messages()
        return {
            "messages": messages,
            "context": context,
            "from_cache": False,
            "cached_answer": None,
            "memory_context": memory_context,
        }

    def _merge_context(self, memory_context: str, retrieval_context: str) -> str:
        sections = []
        if memory_context.strip():
            sections.append(memory_context.strip())
        if retrieval_context.strip():
            sections.append(retrieval_context.strip())
        if not sections:
            return ""
        separator = self.retrieval.context_separator or "\n\n"
        return separator.join(sections)

    def run_cli(self) -> None:
        self.console.rule(
            f"RAG + {self.provider_name.title()} chat. Press Enter on an empty line or Ctrl+D to exit."
        )
        while True:
            try:
                query = self.console.input("[prompt]\nYou:[/prompt] ").strip()
            except EOFError:
                self.console.print()
                break
            if not query:
                break
            with self.console.status("[info]Preparing augmented context...[/info]"):
                prepared = self._prepare_prompt(query)

            if prepared["from_cache"]:
                answer = prepared["cached_answer"] or ""
                if self.memory:
                    self.memory.remember(query, answer, from_cache=True)
            else:
                with self.console.status(f"[info]Calling {self.provider_name}...[/info]"):
                    try:
                        answer = self.client.chat(prepared["messages"])
                    except ModelClientError as exc:
                        self.console.print(
                            Panel(
                                f"Encountered a {self.provider_name} error: {exc}",
                                title="Provider Error",
                                style="error",
                            )
                        )
                        break
                if self.memory:
                    self.memory.remember(query, answer)
            self.console.print(Panel.fit("Assistant:", style="bold"))
            if prepared["from_cache"]:
                self.console.print(
                    Panel(
                        "Response served from persistent memory.",
                        title="Memory Hit",
                        style="info",
                    )
                )
            self.console.print(Markdown(answer) if answer else "[warning]No response received.[/warning]")
