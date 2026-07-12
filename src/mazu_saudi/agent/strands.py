"""AWS Strands warning orchestration wrapper and deterministic helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mazu_saudi.agent.workflow import SaudiWarningPipeline
from mazu_saudi.config import StrandsSettings

Context = dict[str, Any]


class StrandsError(RuntimeError):
    """Raised when Strands orchestration cannot execute."""


class WarningRequestArgs(BaseModel):
    """Strict input contract for warning generation requests."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any]
    industries: list[str] | None = None
    language: str = Field(default="zh", min_length=2, max_length=8, pattern=r"^[a-z_]+$")


class PipelineToolArgs(BaseModel):
    """Validated arguments for the deterministic pipeline tool."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any]


class ResponseAssemblyArgs(BaseModel):
    """Validated arguments for response assembly."""

    model_config = ConfigDict(extra="forbid")

    publish_payload: dict[str, Any]
    industries: list[str] | None = None
    language: str = Field(default="zh", min_length=2, max_length=8, pattern=r"^[a-z_]+$")


@dataclass
class StrandsRun:
    """Structured execution metadata for one Strands-backed run."""

    run_id: str
    status: str
    execution_mode: str
    provider: str
    model_id: str
    region: str
    agent_name: str
    system_prompt: str
    duration_ms: float
    tool_steps: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "execution_mode": self.execution_mode,
            "provider": self.provider,
            "model_id": self.model_id,
            "region": self.region,
            "agent_name": self.agent_name,
            "system_prompt": self.system_prompt,
            "duration_ms": self.duration_ms,
            "tool_steps": self.tool_steps,
            "summary": self.summary,
            "raw_response": self.raw_response,
            "error": self.error,
        }


@dataclass
class StrandsExecutionResult:
    """Combined warning output and execution context."""

    output: dict[str, Any]
    context: Context
    strands_run: StrandsRun


def build_strands_run_metadata(
    *,
    settings: StrandsSettings,
    started_at: float,
    tool_steps: list[dict[str, Any]],
    status: str,
    execution_mode: str,
    summary: str,
    raw_response: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> StrandsRun:
    """Build the structured run metadata exposed by the API."""

    return StrandsRun(
        run_id=f"strands-{uuid.uuid4().hex[:12]}",
        status=status,
        execution_mode=execution_mode,
        provider=settings.provider,
        model_id=settings.model_id,
        region=settings.region,
        agent_name=settings.agent_name,
        system_prompt=settings.system_prompt,
        duration_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
        tool_steps=tool_steps,
        summary=summary,
        raw_response=raw_response or {},
        error=error or {},
    )


def _pipeline_error_message(context: Context) -> str:
    errors = context.get("errors", [])
    if not errors:
        return "warning pipeline failed"
    return str(errors[-1].get("message", "warning pipeline failed"))


def _build_warning_output(
    *,
    publish_payload: dict[str, Any],
    requested_industries: list[str] | None,
    language: str,
    strands_run: dict[str, Any],
) -> dict[str, Any]:
    output = dict(publish_payload)
    if requested_industries:
        output["briefings"] = [briefing for briefing in output.get("briefings", []) if briefing.get("industry") in requested_industries]
    output["requested_language"] = language
    output["briefing_text"] = [
        {"industry": item["industry"], "text": item.get(language, item["zh"])}
        for item in output.get("briefings", [])
    ]
    output["llm_raw"] = output.get("llm_raw", {})
    output["generation_metadata"] = output.get("generation_metadata", {})
    output["strands_run"] = strands_run
    return output


def _run_pipeline_or_raise(features: dict[str, Any], pipeline_factory: Callable[[], SaudiWarningPipeline]) -> Context:
    context = pipeline_factory().run(features)
    if context.get("failed"):
        raise RuntimeError(_pipeline_error_message(context))
    return context


class StrandsWarningAgent:
    """Strands wrapper around the warning pipeline."""

    def __init__(
        self,
        settings: StrandsSettings | None = None,
        agent_runner: Callable[["StrandsWarningAgent", WarningRequestArgs, list[dict[str, Any]]], tuple[Context, dict[str, Any]]] | None = None,
        pipeline_factory: Callable[[], SaudiWarningPipeline] | None = None,
        sdk_agent_factory: Callable[..., Any] | None = None,
    ):
        self.settings = settings or StrandsSettings.from_env()
        self.agent_runner = agent_runner
        self.pipeline_factory = pipeline_factory or SaudiWarningPipeline
        self.sdk_agent_factory = sdk_agent_factory

    def _ensure_runtime_ready(self) -> None:
        if self.agent_runner is not None:
            return
        if importlib.util.find_spec("strands") is None and importlib.util.find_spec("strands_agents") is None:
            raise StrandsError("Strands SDK is not installed. Install dependencies with the official strands-agents package.")
        if not self.settings.enabled:
            raise StrandsError("Strands runtime was invoked while disabled.")
        if self.settings.provider != "bedrock":
            raise StrandsError(f"Unsupported Strands provider '{self.settings.provider}'. Only 'bedrock' is supported.")

    def _load_sdk_symbols(self) -> tuple[Any, Any, Any]:
        try:
            from strands import Agent, tool
            from strands.models import BedrockModel
        except Exception as exc:  # pragma: no cover - import failure depends on environment
            raise StrandsError(f"Failed to import Strands SDK: {exc}") from exc
        return Agent, BedrockModel, tool

    def _build_sdk_error(self, exc: Exception) -> StrandsError:
        message = str(exc).strip() or exc.__class__.__name__
        exception_name = exc.__class__.__name__
        lowered = message.lower()
        if exception_name in {"NoCredentialsError", "PartialCredentialsError", "CredentialRetrievalError"}:
            return StrandsError(f"AWS credentials are not configured for Strands Bedrock execution: {message}")
        if exception_name in {"ProxyConnectionError", "ConnectTimeoutError", "EndpointConnectionError"}:
            return StrandsError(f"Unable to reach AWS Bedrock for Strands execution: {message}")
        if "unable to locate credentials" in lowered or "credential" in lowered and "aws" in lowered:
            return StrandsError(f"AWS credentials are not configured for Strands Bedrock execution: {message}")
        if "accessdenied" in lowered or "not authorized" in lowered or "access denied" in lowered:
            return StrandsError(f"AWS Bedrock access was denied for Strands execution: {message}")
        if "validationexception" in lowered or "model identifier is invalid" in lowered or "model" in lowered and "invalid" in lowered:
            return StrandsError(f"Invalid Strands Bedrock model configuration: {message}")
        return StrandsError(f"Strands SDK execution failed: {message}")

    def _record_tool_step(self, name: str, status: str, started_at: float, error: Exception | None = None) -> dict[str, Any]:
        return {
            "name": name,
            "status": status,
            "duration_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            **({"error": {"type": error.__class__.__name__, "message": str(error)}} if error is not None else {}),
        }

    def _tool_validate_request(self, args: WarningRequestArgs) -> WarningRequestArgs:
        return args

    def _tool_run_pipeline(self, args: PipelineToolArgs) -> Context:
        return _run_pipeline_or_raise(args.features, self.pipeline_factory)

    def _tool_assemble_response(self, args: ResponseAssemblyArgs) -> dict[str, Any]:
        return _build_warning_output(
            publish_payload=args.publish_payload,
            requested_industries=args.industries,
            language=args.language,
            strands_run={},
        )

    def _run_fixed_sop(
        self,
        args: WarningRequestArgs,
        tool_steps: list[dict[str, Any]],
    ) -> tuple[Context, dict[str, Any]]:
        step_started_at = time.perf_counter()
        validated = self._tool_validate_request(args)
        tool_steps.append(self._record_tool_step("validate_request", "ok", step_started_at))

        step_started_at = time.perf_counter()
        try:
            context = self._tool_run_pipeline(PipelineToolArgs(features=validated.features))
        except Exception as exc:
            tool_steps.append(self._record_tool_step("run_warning_pipeline", "error", step_started_at, exc))
            raise
        tool_steps.append(self._record_tool_step("run_warning_pipeline", "ok", step_started_at))

        step_started_at = time.perf_counter()
        output = self._tool_assemble_response(
            ResponseAssemblyArgs(
                publish_payload=context["publish_payload"],
                industries=validated.industries,
                language=validated.language,
            )
        )
        tool_steps.append(self._record_tool_step("assemble_response", "ok", step_started_at))
        return context, output

    def _run_with_strands_sdk(
        self,
        args: WarningRequestArgs,
        tool_steps: list[dict[str, Any]],
    ) -> tuple[Context, dict[str, Any], dict[str, Any]]:
        Agent, BedrockModel, tool = self._load_sdk_symbols()
        os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

        sdk_state: dict[str, Any] = {"context": None, "output": None}

        @tool(name="validate_request", description="Validate the warning request payload.")
        def validate_request(features: dict[str, Any], industries: list[str] | None = None, language: str = "zh") -> dict[str, Any]:
            step_started_at = time.perf_counter()
            try:
                validated = self._tool_validate_request(
                    WarningRequestArgs(features=features, industries=industries, language=language)
                )
            except Exception as exc:
                tool_steps.append(self._record_tool_step("validate_request", "error", step_started_at, exc))
                raise
            tool_steps.append(self._record_tool_step("validate_request", "ok", step_started_at))
            return validated.model_dump()

        @tool(name="run_warning_pipeline", description="Run the MAZU Saudi warning pipeline and return the publish payload.")
        def run_warning_pipeline(features: dict[str, Any]) -> dict[str, Any]:
            step_started_at = time.perf_counter()
            try:
                context = self._tool_run_pipeline(PipelineToolArgs(features=features))
            except Exception as exc:
                tool_steps.append(self._record_tool_step("run_warning_pipeline", "error", step_started_at, exc))
                raise
            sdk_state["context"] = context
            tool_steps.append(self._record_tool_step("run_warning_pipeline", "ok", step_started_at))
            return {"publish_payload": context["publish_payload"], "risk_count": len(context.get("risks", []))}

        @tool(name="assemble_response", description="Assemble the final warning API response payload.")
        def assemble_response(publish_payload: dict[str, Any], industries: list[str] | None = None, language: str = "zh") -> dict[str, Any]:
            step_started_at = time.perf_counter()
            try:
                output = self._tool_assemble_response(
                    ResponseAssemblyArgs(
                        publish_payload=publish_payload,
                        industries=industries,
                        language=language,
                    )
                )
            except Exception as exc:
                tool_steps.append(self._record_tool_step("assemble_response", "error", step_started_at, exc))
                raise
            sdk_state["output"] = output
            tool_steps.append(self._record_tool_step("assemble_response", "ok", step_started_at))
            return output

        prompt = (
            "Execute the workflow exactly once using tools only.\n"
            "1. Call validate_request with the provided payload.\n"
            "2. Call run_warning_pipeline using the validated features.\n"
            "3. Call assemble_response using the pipeline publish_payload plus the original industries and language.\n"
            "Do not retry. Do not skip tools. After the final tool call, return the assembled JSON payload only.\n\n"
            f"payload={json.dumps(args.model_dump(), ensure_ascii=False)}"
        )

        try:
            model = BedrockModel(
                region_name=self.settings.region,
                model_id=self.settings.model_id,
                **self.settings.extra_config,
            )
            agent = (
                self.sdk_agent_factory(model=model, tools=[validate_request, run_warning_pipeline, assemble_response], system_prompt=self.settings.system_prompt, name=self.settings.agent_name)
                if self.sdk_agent_factory is not None
                else Agent(
                    model=model,
                    tools=[validate_request, run_warning_pipeline, assemble_response],
                    system_prompt=self.settings.system_prompt,
                    name=self.settings.agent_name,
                )
            )
            result = agent(prompt)
        except Exception as exc:
            raise self._build_sdk_error(exc) from exc

        if sdk_state["context"] is None or sdk_state["output"] is None:
            raise StrandsError("Strands SDK completed without producing a final assembled response.")

        raw_response = {
            "stop_reason": getattr(result, "stop_reason", None),
            "text": str(result).strip(),
            "message": getattr(result, "message", {}),
        }
        return sdk_state["context"], sdk_state["output"], raw_response

    def execute(self, payload: dict[str, Any] | WarningRequestArgs) -> StrandsExecutionResult:
        """Execute one Strands-backed warning run."""

        started_at = time.perf_counter()
        tool_steps: list[dict[str, Any]] = []
        args = WarningRequestArgs.model_validate(payload)
        execution_mode = "sdk"
        raw_response: dict[str, Any] = {}
        try:
            self._ensure_runtime_ready()
            if self.agent_runner is not None:
                execution_mode = "fixed_sop"
                context, output = self.agent_runner(self, args, tool_steps)
            else:
                context, output, raw_response = self._run_with_strands_sdk(args, tool_steps)
        except StrandsError as exc:
            raise StrandsError(str(exc)) from exc
        except ValidationError:
            raise
        strands_run = build_strands_run_metadata(
            settings=self.settings,
            started_at=started_at,
            tool_steps=tool_steps,
            status="completed",
            execution_mode=execution_mode,
            summary=(
                f"Generated warning output via {execution_mode} Strands orchestration with "
                f"{len(output.get('briefings', []))} briefings and {len(context.get('risks', []))} risk assessments."
            ),
            raw_response=raw_response,
        )
        output["strands_run"] = strands_run.to_dict()
        return StrandsExecutionResult(output=output, context=context, strands_run=strands_run)


def generate_warning_response(
    payload: dict[str, Any] | WarningRequestArgs,
    *,
    settings: StrandsSettings | None = None,
    agent_factory: Callable[..., StrandsWarningAgent] = StrandsWarningAgent,
    pipeline_factory: Callable[[], SaudiWarningPipeline] | None = None,
) -> dict[str, Any]:
    """Generate a warning response through Strands or the deterministic fallback path."""

    settings = settings or StrandsSettings.from_env()
    args = WarningRequestArgs.model_validate(payload)
    if settings.enabled:
        result = agent_factory(settings=settings, pipeline_factory=pipeline_factory).execute(args)
        return result.output
    context = _run_pipeline_or_raise(args.features, pipeline_factory or SaudiWarningPipeline)
    return _build_warning_output(
        publish_payload=context["publish_payload"],
        requested_industries=args.industries,
        language=args.language,
        strands_run={},
    )
