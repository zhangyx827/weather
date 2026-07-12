"""Tests for briefing generation and Strands orchestration helpers."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from mazu_saudi.agent.briefing import (
    BriefingGenerationError,
    OpenAICompatibleBriefingGenerator,
    TemplateBriefingGenerator,
)
from mazu_saudi.agent.strands import StrandsError, StrandsWarningAgent, generate_warning_response
from mazu_saudi.agent.workflow import load_sample_features
from mazu_saudi.config import LLMSettings, StrandsSettings
from mazu_saudi.schemas import GridCell, HazardRisk, RiskLevel


def sample_risks() -> list[HazardRisk]:
    grid = GridCell(id="riyadh-1", lat=24.7, lon=46.7, region="Riyadh")
    return [
        HazardRisk(
            hazard_type="extreme_heat",
            risk_probability=0.82,
            risk_level=RiskLevel.EXTREME,
            contributing_factors=["high_temperature", "low_humidity"],
            grid=grid,
            model_family="lightgbm",
            inference_mode="rule",
        )
    ]


def sample_briefing_payload() -> dict[str, object]:
    return {
        "briefings": [
            {
                "industry": industry,
                "zh": f"{industry} 中文简报",
                "en": f"{industry} english briefing",
                "ar": f"{industry} arabic briefing",
                "hazards": ["extreme_heat"],
            }
            for industry in ("meteorology", "emergency", "agriculture", "transport", "port", "public_health")
        ]
    }


class BriefingGeneratorTests(unittest.TestCase):
    def test_template_generator_returns_all_industries(self):
        result = TemplateBriefingGenerator().generate("Riyadh", sample_risks(), {"impacts": {}})
        self.assertEqual(len(result.briefings), 6)
        self.assertEqual(result.metadata["provider"], "template")

    def test_openai_generator_parses_structured_json(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(sample_briefing_payload(), ensure_ascii=False)
                            }
                        }
                    ]
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        generator = OpenAICompatibleBriefingGenerator(
            settings=LLMSettings(base_url="https://llm.example.com", api_key="secret", model="gpt-test"),
            http_client=client,
        )
        result = generator.generate("Riyadh", sample_risks(), {"impacts": {}}, context={"human_review": {"status": "pending"}})
        self.assertEqual(len(result.briefings), 6)
        self.assertEqual(result.metadata["provider"], "llm")
        self.assertEqual(captured["headers"]["authorization"], "Bearer secret")
        self.assertEqual(captured["body"]["model"], "gpt-test")

    def test_openai_generator_rejects_missing_industries(self):
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps({"briefings": []})}}]},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        generator = OpenAICompatibleBriefingGenerator(
            settings=LLMSettings(base_url="https://llm.example.com", api_key="secret", model="gpt-test"),
            http_client=client,
        )
        with self.assertRaises(BriefingGenerationError):
            generator.generate("Riyadh", sample_risks(), {"impacts": {}})


class StrandsRuntimeTests(unittest.TestCase):
    def test_strands_settings_ignore_legacy_stride_env(self):
        with patch.dict(
            os.environ,
            {
                "MAZU_STRIDE_ENABLED": "true",
                "MAZU_STRIDE_MODEL_ID": "legacy-model",
                "MAZU_STRANDS_ENABLED": "false",
                "MAZU_STRANDS_MODEL_ID": "modern-model",
            },
            clear=False,
        ):
            settings = StrandsSettings.from_env()
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.model_id, "modern-model")

    def test_agent_validation_blocks_invalid_input_before_pipeline(self):
        called = {"pipeline": False}

        class ExplodingPipeline:
            def run(self, features):
                called["pipeline"] = True
                raise AssertionError("pipeline should not run")

        agent = StrandsWarningAgent(
            settings=StrandsSettings(enabled=True),
            agent_runner=lambda runtime, args, steps: runtime._run_fixed_sop(args, steps),
            pipeline_factory=ExplodingPipeline,
        )
        with self.assertRaises(ValidationError):
            agent.execute({"features": "not-a-dict", "language": "zh"})
        self.assertFalse(called["pipeline"])

    def test_enabled_and_disabled_paths_produce_identical_warning_content(self):
        payload = {
            "features": load_sample_features(),
            "industries": ["meteorology"],
            "language": "zh",
        }
        enabled_output = StrandsWarningAgent(
            settings=StrandsSettings(enabled=True),
            agent_runner=lambda runtime, args, steps: runtime._run_fixed_sop(args, steps),
        ).execute(payload).output
        disabled_output = generate_warning_response(
            payload,
            settings=StrandsSettings(enabled=False),
        )
        self.assertEqual(enabled_output["generation_metadata"], disabled_output["generation_metadata"])
        self.assertEqual(enabled_output["briefing_text"], disabled_output["briefing_text"])
        self.assertTrue(enabled_output["strands_run"])
        self.assertEqual(disabled_output["strands_run"], {})
        self.assertEqual(enabled_output["strands_run"]["execution_mode"], "fixed_sop")

    def test_enabled_sdk_path_uses_agent_tools_and_records_raw_response(self):
        calls = []
        from strands import tool as strands_tool

        class FakeBedrockModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeResult:
            stop_reason = "end_turn"
            message = {"role": "assistant", "content": [{"text": "{\"status\":\"ok\"}"}]}

            def __str__(self):
                return "{\"status\":\"ok\"}"

        class FakeSDKAgent:
            def __init__(self, *, model, tools, system_prompt, name):
                self.model = model
                self.tools = {tool.tool_spec["name"]: tool for tool in tools}
                self.system_prompt = system_prompt
                self.name = name

            def __call__(self, prompt):
                self.tools["validate_request"](features=load_sample_features(), industries=["meteorology"], language="zh")
                pipeline = self.tools["run_warning_pipeline"](features=load_sample_features())
                self.tools["assemble_response"](
                    publish_payload=pipeline["publish_payload"],
                    industries=["meteorology"],
                    language="zh",
                )
                calls.append(prompt)
                return FakeResult()

        agent = StrandsWarningAgent(
            settings=StrandsSettings(enabled=True, model_id="amazon.nova-lite-v1:0", region="us-east-1"),
            sdk_agent_factory=FakeSDKAgent,
        )
        with patch.object(agent, "_load_sdk_symbols", return_value=(None, FakeBedrockModel, strands_tool)):
            output = agent.execute(
                {"features": load_sample_features(), "industries": ["meteorology"], "language": "zh"}
            ).output

        self.assertEqual(output["strands_run"]["execution_mode"], "sdk")
        self.assertEqual(output["strands_run"]["raw_response"]["stop_reason"], "end_turn")
        self.assertEqual([step["name"] for step in output["strands_run"]["tool_steps"]], ["validate_request", "run_warning_pipeline", "assemble_response"])
        self.assertTrue(calls)

    def test_enabled_sdk_path_maps_bedrock_errors(self):
        from strands import tool as strands_tool

        class FakeBedrockModel:
            def __init__(self, **kwargs):
                raise RuntimeError("Unable to locate credentials for AWS request signing")

        agent = StrandsWarningAgent(
            settings=StrandsSettings(enabled=True),
            sdk_agent_factory=lambda **kwargs: kwargs,
        )
        with patch.object(agent, "_load_sdk_symbols", return_value=(None, FakeBedrockModel, strands_tool)):
            with self.assertRaises(StrandsError) as exc:
                agent.execute({"features": load_sample_features(), "language": "zh"})
        self.assertIn("AWS credentials", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
