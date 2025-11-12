"""High-level chat orchestration for the RAG demo."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Sequence

from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import PromptConfig, RetrievalConfig
from .console import console
from .llm import ModelClientError, ModelInfo, ModelNotFoundError, SupportsModels
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
        console_override=None,
    ) -> None:
        self.client = client
        self.context_builder = context_builder
        self.retrieval = retrieval
        self.prompts = prompts
        self.provider_name = provider_name
        self.console = console_override or console

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

    def build_context(self, query: str, *, top_k: int | None = None, max_chars: int | None = None) -> str:
        return self.context_builder.build_context(query, top_k=top_k, max_chars=max_chars)

    def ask(self, query: str) -> str:
        context = self.build_context(query)
        messages = ChatRequest(query=query, context=context, prompts=self.prompts).to_messages()
        return self.client.chat(messages)

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
            with self.console.status("[info]Retrieving context from the vector store...[/info]"):
                context = self.build_context(query)
            with self.console.status(f"[info]Calling {self.provider_name}...[/info]"):
                try:
                    answer = self.client.chat(
                        ChatRequest(query=query, context=context, prompts=self.prompts).to_messages()
                    )
                except ModelClientError as exc:
                    self.console.print(
                        Panel(
                            f"Encountered a {self.provider_name} error: {exc}",
                            title="Provider Error",
                            style="error",
                        )
                    )
                    break
            self.console.print(Panel.fit("Assistant:", style="bold"))
            self.console.print(Markdown(answer) if answer else "[warning]No response received.[/warning]")
