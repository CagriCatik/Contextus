"""Generate structured documentation such as test specifications via the RAG pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Sequence

from .config import DocumentationConfig, RetrievalConfig
from .llm import SupportsModels
from .retrieval import ContextBuilder


@dataclass
class TestCase:
    """Structured representation of a generated test case."""

    test_case_id: str
    test_case_name: str
    referenced_requirements: List[str]
    precondition: str
    action: str
    expected_result: str
    postcondition: str
    raw_response: str | None = None

    def to_markdown(self) -> str:
        """Render the test case as a Markdown section."""

        refs = ", ".join(self.referenced_requirements) if self.referenced_requirements else "-"
        lines = [
            f"### {self.test_case_id}: {self.test_case_name}",
            "",
            f"**Referenced requirements:** {refs}",
            "",
            f"**Precondition:** {self.precondition}",
            "",
            f"**Action:** {self.action}",
            "",
            f"**Expected result:** {self.expected_result}",
            "",
            f"**Postcondition:** {self.postcondition}",
        ]
        if self.raw_response:
            lines.extend([
                "",
                "<details><summary>Raw provider response</summary>",
                "",
                "```",
                self.raw_response,
                "```",
                "</details>",
            ])
        return "\n".join(lines)


class TestSpecGenerator:
    """Coordinate context retrieval and LLM prompting to build test specifications."""

    def __init__(
        self,
        client: SupportsModels,
        context_builder: ContextBuilder,
        documentation: DocumentationConfig,
        retrieval: RetrievalConfig,
    ) -> None:
        self.client = client
        self.context_builder = context_builder
        self.documentation = documentation
        self.retrieval = retrieval

    def generate_test_case(
        self,
        requirement_query: str,
        *,
        ordinal: int,
        top_k: int | None = None,
        max_context_chars: int | None = None,
    ) -> TestCase:
        """Generate a single structured test case for the given requirement query."""

        test_case_id = self._format_identifier(ordinal)
        context = self.context_builder.build_context(
            requirement_query,
            top_k=top_k or self.retrieval.top_k,
            max_chars=max_context_chars or self.retrieval.max_context_chars,
        )
        messages = self._build_messages(requirement_query, test_case_id, context)
        response = self.client.chat(messages)
        return self._parse_response(response, test_case_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _format_identifier(self, ordinal: int) -> str:
        prefix = self.documentation.test_case_id_prefix
        return f"{prefix}{ordinal:03d}" if prefix else f"TC-{ordinal:03d}"

    def _build_messages(self, requirement_query: str, test_case_id: str, context: str) -> Sequence[dict]:
        request_block = (
            "You must craft a single manual test case that validates the specified requirements.\n"
            "Respond with a JSON object using the following schema:\n"
            "{\n"
            f"  \"test_case_id\": \"{test_case_id}\",\n"
            "  \"test_case_name\": string,\n"
            "  \"referenced_requirements\": [list of requirement identifiers extracted from the context],\n"
            "  \"precondition\": string,\n"
            "  \"action\": string detailing tester steps,\n"
            "  \"expected_result\": string,\n"
            "  \"postcondition\": string\n"
            "}\n"
            "Use the requirement query and the retrieved context below.\n"
            "Do not include any explanatory text outside of the JSON payload."
        )
        user_payload = (
            f"Requirement query: {requirement_query}\n\n"
            "Retrieved requirement context:\n"
            f"{context if context.strip() else '[no matching context found]'}"
        )
        return [
            {"role": "system", "content": self.documentation.system_prompt},
            {"role": "user", "content": request_block},
            {"role": "user", "content": user_payload},
        ]

    def _parse_response(self, response: str, fallback_id: str) -> TestCase:
        cleaned = self._strip_code_fences(response.strip())
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
            return TestCase(
                test_case_id=fallback_id,
                test_case_name="Unparsed response",
                referenced_requirements=[],
                precondition="",
                action="",
                expected_result="",
                postcondition="",
                raw_response=response.strip(),
            )

        referenced = payload.get("referenced_requirements")
        if isinstance(referenced, str):
            referenced_requirements = [referenced]
        elif isinstance(referenced, list):
            referenced_requirements = [str(item) for item in referenced]
        else:
            referenced_requirements = []

        return TestCase(
            test_case_id=str(payload.get("test_case_id", fallback_id)),
            test_case_name=str(payload.get("test_case_name", "Unnamed Test Case")),
            referenced_requirements=referenced_requirements,
            precondition=str(payload.get("precondition", "")),
            action=str(payload.get("action", "")),
            expected_result=str(payload.get("expected_result", "")),
            postcondition=str(payload.get("postcondition", "")),
            raw_response=None,
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                # drop first fence
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return text
