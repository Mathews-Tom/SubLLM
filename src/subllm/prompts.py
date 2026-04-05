"""Prompt registry and versioned prompt resolution for SubLLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from subllm.errors import PromptRenderError, UnknownPromptError


@dataclass(frozen=True)
class RegisteredPrompt:
    name: str
    version: str
    template: str
    description: str | None = None
    variables: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedPrompt:
    name: str
    version: str
    text: str
    metadata: dict[str, str]


class PromptRegistry:
    def __init__(self) -> None:
        self._prompts: dict[str, dict[str, RegisteredPrompt]] = {}
        self._defaults: dict[str, str] = {}

    def register(
        self,
        *,
        name: str,
        version: str,
        template: str,
        description: str | None = None,
        variables: tuple[str, ...] = (),
        metadata: dict[str, str] | None = None,
        default: bool = False,
    ) -> None:
        prompt = RegisteredPrompt(
            name=name,
            version=version,
            template=template,
            description=description,
            variables=variables,
            metadata=metadata or {},
        )
        versions = self._prompts.setdefault(name, {})
        versions[version] = prompt
        if default or name not in self._defaults:
            self._defaults[name] = version

    def list_prompts(self) -> list[RegisteredPrompt]:
        prompts = [prompt for versions in self._prompts.values() for prompt in versions.values()]
        return sorted(prompts, key=lambda prompt: (prompt.name, prompt.version))

    def resolve(
        self,
        *,
        name: str,
        version: str | None = None,
        variables: dict[str, str] | None = None,
    ) -> ResolvedPrompt:
        versions = self._prompts.get(name)
        if versions is None:
            raise UnknownPromptError(
                prompt_name=name,
                detail=f"Unknown prompt '{name}'. Register it before use.",
            )

        resolved_version = version or self._defaults.get(name)
        if resolved_version is None:
            raise UnknownPromptError(
                prompt_name=name,
                detail=f"Prompt '{name}' requires an explicit version.",
            )

        prompt = versions.get(resolved_version)
        if prompt is None:
            raise UnknownPromptError(
                prompt_name=name,
                detail=(
                    f"Unknown prompt version '{resolved_version}' for '{name}'. "
                    f"Available versions: {', '.join(sorted(versions))}"
                ),
            )

        resolved_variables = variables or {}
        expected_variables = set(prompt.variables)
        actual_variables = set(resolved_variables)

        missing_variables = sorted(expected_variables - actual_variables)
        if missing_variables:
            raise PromptRenderError(
                message=(
                    f"Prompt '{name}' version '{prompt.version}' is missing variables: "
                    f"{', '.join(missing_variables)}"
                )
            )

        unexpected_variables = sorted(actual_variables - expected_variables)
        if unexpected_variables:
            raise PromptRenderError(
                message=(
                    f"Prompt '{name}' version '{prompt.version}' received unsupported variables: "
                    f"{', '.join(unexpected_variables)}"
                )
            )

        try:
            text = prompt.template.format_map(resolved_variables)
        except KeyError as exc:
            raise PromptRenderError(
                message=(
                    f"Prompt '{name}' version '{prompt.version}' could not render variable "
                    f"'{exc.args[0]}'"
                )
            ) from exc

        metadata = {
            "name": prompt.name,
            "version": prompt.version,
            **prompt.metadata,
        }
        return ResolvedPrompt(
            name=prompt.name,
            version=prompt.version,
            text=text,
            metadata=metadata,
        )


_PROMPT_REGISTRY = PromptRegistry()
_PROMPT_REGISTRY.register(
    name="chat-default",
    version="v1",
    template="You are a precise, direct assistant. Answer with clear technical statements.",
    description="General-purpose assistant baseline for deterministic replies.",
    metadata={"category": "general"},
    default=True,
)
_PROMPT_REGISTRY.register(
    name="code-review",
    version="v1",
    template=(
        "Review the code with a strict engineering lens. "
        "Prioritize bugs, regressions, and missing tests."
    ),
    description="Review-focused system prompt for findings-first output.",
    metadata={"category": "engineering"},
    default=True,
)
_PROMPT_REGISTRY.register(
    name="release-notes",
    version="v1",
    template=(
        "Summarize changes for {audience}. "
        "Keep the output concise, concrete, and organized by user impact."
    ),
    description="Release-summary prompt with an audience variable.",
    variables=("audience",),
    metadata={"category": "documentation"},
    default=True,
)


def get_prompt_registry() -> PromptRegistry:
    return _PROMPT_REGISTRY


def list_registered_prompts() -> list[RegisteredPrompt]:
    return get_prompt_registry().list_prompts()


def resolve_prompt(
    *,
    name: str,
    version: str | None = None,
    variables: dict[str, str] | None = None,
) -> ResolvedPrompt:
    return get_prompt_registry().resolve(name=name, version=version, variables=variables)
