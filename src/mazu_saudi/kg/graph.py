"""RDF/OWL-style knowledge graph minimum closed loop."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from mazu_saudi.schemas import HazardRisk

MAZU = "https://mazu.example.org/saudi#"
SAKUNA = "https://sakuna.ph/"
BAWARE = "https://raw.githubusercontent.com/beAWARE-project/ontology/master/beAWARE_ontology#"


@dataclass(frozen=True)
class OntologyConfig:
    """Central URI config for the MAZU Saudi ontology namespace."""

    base_uri: str = MAZU
    geosparql_uri: str = "http://www.opengis.net/ont/geosparql#"
    shacl_uri: str = "http://www.w3.org/ns/shacl#"


def load_hazard_rules(path: str | Path | None = None) -> dict[str, Any]:
    """Load hazard-exposure-department-action rules from JSON config."""

    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "hazard_response_rules.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


DEFAULT_HAZARD_KNOWLEDGE = load_hazard_rules()


def validate_instance_ttl(payload: str) -> dict[str, Any]:
    """Check the grounding persistence contract in an instance Turtle export."""

    required_markers = ("mazu:GroundingGap", "mazu:hasGroundingGap", "mazu:groundingPayload")
    missing = [marker for marker in required_markers if marker not in payload]
    return {
        "valid": not missing,
        "missing_markers": missing,
        "bytes": len(payload.encode("utf-8")),
    }


@dataclass(frozen=True)
class Triple:
    """Simple RDF-like triple."""

    subject: str
    predicate: str
    object: str

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


class HazardKnowledgeGraph:
    """SakunaGraPH-backed RDF graph with MAZU operational query facade."""

    def __init__(self, ontology: OntologyConfig | None = None, rules: dict[str, Any] | None = None, backend: str | None = None) -> None:
        self.ontology = ontology or OntologyConfig()
        self.rules = rules or DEFAULT_HAZARD_KNOWLEDGE
        self.namespace = self.ontology.base_uri
        self.triples: list[Triple] = []
        self._rdflib_graph = None
        self.sakuna_adapter = SakunaGraphAdapter()
        self.backend = (backend or os.getenv("MAZU_KG_BACKEND", "sakuna")).strip().lower()
        self.backend_metadata: dict[str, Any] = {
            "backend": "local_rdflib",
            "status": "initialized",
            "ontology_triple_count": 0,
        }
        try:
            from rdflib import Graph, Namespace, URIRef

            if self.backend in {"sakuna", "sakunagraph"} and self.sakuna_adapter.available():
                self._rdflib_graph = self.sakuna_adapter.load_graph()
                self.backend_metadata = self.sakuna_adapter.metadata()
            else:
                self._rdflib_graph = Graph()
            self._namespace = Namespace(self.namespace)
            self._rdflib_graph.bind("mazu", self._namespace)
            self._rdflib_graph.bind("sakuna", Namespace(SAKUNA))
            self._rdflib_graph.bind("geo", Namespace(self.ontology.geosparql_uri))
            self._rdflib_graph.bind("sh", Namespace(self.ontology.shacl_uri))
            self._rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            self._rdfs_label = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
        except Exception:
            self._namespace = None
            self.backend_metadata = {
                "backend": "local_triples",
                "status": "rdflib_unavailable",
                "ontology_triple_count": 0,
            }
        self._seed_ontology()

    @staticmethod
    def _slug(value: Any) -> str:
        text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "unknown")).strip("_")
        return text or "unknown"

    @staticmethod
    def _digest(payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        return sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _valid_time_text(risk: HazardRisk) -> str:
        value = risk.valid_time
        if isinstance(value, datetime):
            return value.isoformat()
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _top_shap_features(risk: HazardRisk) -> list[dict[str, Any]]:
        items = risk.shap_summary.get("top_features", [])
        if not isinstance(items, list):
            return []
        normalized = []
        for item in items[:5]:
            if isinstance(item, dict):
                normalized.append(dict(item))
            else:
                normalized.append({"feature": str(item)})
        return normalized

    def _risk_event_uri(self, risk: HazardRisk) -> str:
        grid_id = self._slug(risk.grid.id if risk.grid else "unknown")
        event_key = {
            "hazard_type": risk.hazard_type,
            "grid_id": grid_id,
            "valid_time": self._valid_time_text(risk),
            "model_name": risk.model_name,
        }
        return f"{self.namespace}event_{self._slug(risk.hazard_type)}_{grid_id}_{self._digest(event_key)}"

    def _seed_ontology(self) -> None:
        classes = [
            "MeteorologicalIndicator",
            "HazardScenario",
            "ExposureObject",
            "ServiceDepartment",
            "WarningProduct",
            "ResponseAction",
        ]
        for cls in classes:
            self.add(MAZU + cls, "rdf:type", "owl:Class")
        for hazard, info in self.rules.items():
            subject = self.namespace + hazard
            self.add(subject, "rdf:type", self.namespace + "HazardScenario")
            self.add(subject, "rdfs:label", info["label"])
            for exposure in info["exposures"]:
                self.add(subject, self.namespace + "affectsExposure", self.namespace + exposure)
                self.add(self.namespace + exposure, "rdf:type", self.namespace + "ExposureObject")
            for action in info["actions"]:
                self.add(subject, self.namespace + "requiresAction", self.namespace + action)
                self.add(self.namespace + action, "rdf:type", self.namespace + "ResponseAction")
            for department in info["departments"]:
                self.add(subject, self.namespace + "servedByDepartment", self.namespace + department)
                self.add(self.namespace + department, "rdf:type", self.namespace + "ServiceDepartment")
            for product in info.get("warning_products", []):
                self.add(subject, self.namespace + "hasWarningProduct", self.namespace + product)
                self.add(self.namespace + product, "rdf:type", self.namespace + "WarningProduct")

    def add(self, subject: str, predicate: str, obj: str) -> None:
        """Add a triple to the internal graph."""

        triple = Triple(subject, predicate, obj)
        if triple not in self.triples:
            self.triples.append(triple)
        if self._rdflib_graph is not None:
            from rdflib import Literal, URIRef

            s = URIRef(subject) if subject.startswith("http") else URIRef(self.namespace + subject)
            if predicate == "rdf:type":
                p = self._rdf_type
            elif predicate == "rdfs:label":
                p = self._rdfs_label
            else:
                p = URIRef(predicate) if predicate.startswith("http") else URIRef(self.namespace + predicate.replace(":", "_"))
            o = URIRef(obj) if obj.startswith("http") else Literal(obj)
            self._rdflib_graph.add((s, p, o))

    def add_risk_evidence(self, risk: HazardRisk) -> None:
        """Generate triples connecting a risk output to indicators and level."""

        risk_id = f"{self.namespace}risk_{risk.hazard_type}_{risk.grid.id if risk.grid else 'unknown'}"
        hazard_uri = self.namespace + risk.hazard_type
        self.add(risk_id, "rdf:type", self.namespace + "WarningProduct")
        self.add(risk_id, self.namespace + "warnsAbout", hazard_uri)
        self.add(risk_id, self.namespace + "hasRiskLevel", risk.risk_level.value)
        for factor in risk.contributing_factors:
            indicator_id = self.namespace + "indicator_" + str(abs(hash((risk.hazard_type, factor))))
            self.add(indicator_id, "rdf:type", self.namespace + "MeteorologicalIndicator")
            self.add(risk_id, self.namespace + "hasContributingFactor", indicator_id)
            self.add(indicator_id, "rdfs:label", factor)
        for name, value in risk.indicator_evidence.items():
            indicator_id = self.namespace + f"indicator_{risk.hazard_type}_{name}"
            self.add(indicator_id, "rdf:type", self.namespace + "MeteorologicalIndicator")
            self.add(risk_id, self.namespace + "hasIndicatorEvidence", indicator_id)
            self.add(indicator_id, "rdfs:label", name)
            self.add(indicator_id, self.namespace + "hasIndicatorValue", str(value))
        self.add_sakuna_risk_event(risk)

    def add_sakuna_risk_event(self, risk: HazardRisk) -> str:
        """Add SakunaGraPH-compatible instance triples for one MAZU risk output."""

        event_uri = self._risk_event_uri(risk)
        disaster_type_uri = self.sakuna_adapter.disaster_type_uri(risk.hazard_type)
        self.add(event_uri, "rdf:type", SAKUNA + "DisasterEvent")
        self.add(event_uri, SAKUNA + "eventName", f"MAZU {risk.hazard_type} warning")
        self.add(event_uri, SAKUNA + "startDate", self._valid_time_text(risk))
        self.add(event_uri, self.namespace + "riskProbability", f"{float(risk.risk_probability):.6f}")
        self.add(event_uri, self.namespace + "riskLevel", risk.risk_level.value)
        self.add(event_uri, self.namespace + "modelFamily", risk.model_family)
        self.add(event_uri, self.namespace + "modelName", risk.model_name)
        self.add(event_uri, self.namespace + "modelVersion", risk.model_version)
        self.add(event_uri, self.namespace + "inferenceMode", risk.inference_mode)
        self.add(event_uri, self.namespace + "sourceStatus", risk.source_status)
        if disaster_type_uri:
            self.add(event_uri, SAKUNA + "hasDisasterType", disaster_type_uri)
        if risk.grid is not None:
            location_uri = f"{self.namespace}location_{self._slug(risk.grid.id)}"
            self.add(location_uri, "rdf:type", SAKUNA + "Location")
            self.add(location_uri, "rdf:type", "http://www.opengis.net/ont/geosparql#Feature")
            self.add(location_uri, "rdfs:label", risk.grid.region or risk.grid.id)
            self.add(location_uri, self.namespace + "latitude", str(risk.grid.lat))
            self.add(location_uri, self.namespace + "longitude", str(risk.grid.lon))
            self.add(event_uri, SAKUNA + "hasLocation", location_uri)
        for name, value in risk.indicator_evidence.items():
            indicator_uri = f"{event_uri}/indicator/{self._slug(name)}"
            self.add(indicator_uri, "rdf:type", BAWARE + "ClimateParameter")
            self.add(indicator_uri, "rdfs:label", name)
            self.add(indicator_uri, BAWARE + "hasValue", str(value))
            self.add(event_uri, BAWARE + "hasClimateParameterMeasurement", indicator_uri)
        for index, factor in enumerate(risk.contributing_factors):
            factor_uri = f"{event_uri}/factor/{index}_{self._slug(factor)}"
            self.add(factor_uri, "rdf:type", self.namespace + "ModelExplanationFactor")
            self.add(factor_uri, "rdfs:label", factor)
            self.add(event_uri, self.namespace + "hasModelExplanationFactor", factor_uri)
        for index, item in enumerate(self._top_shap_features(risk)):
            feature_name = item.get("feature") or item.get("name") or f"feature_{index}"
            shap_uri = f"{event_uri}/shap/{index}_{self._slug(feature_name)}"
            self.add(shap_uri, "rdf:type", self.namespace + "ShapFeatureEvidence")
            self.add(shap_uri, "rdfs:label", str(feature_name))
            if "value" in item:
                self.add(shap_uri, self.namespace + "shapValue", str(item["value"]))
            if "mean_abs_shap" in item:
                self.add(shap_uri, self.namespace + "meanAbsShap", str(item["mean_abs_shap"]))
            self.add(event_uri, self.namespace + "hasShapFeatureEvidence", shap_uri)
        if risk.degradation_metadata:
            degradation_uri = f"{event_uri}/degradation/{self._digest(risk.degradation_metadata)}"
            self.add(degradation_uri, "rdf:type", self.namespace + "DegradationEvent")
            self.add(degradation_uri, self.namespace + "degradationPayload", json.dumps(risk.degradation_metadata, ensure_ascii=False, sort_keys=True, default=str))
            self.add(event_uri, self.namespace + "hasDegradationEvent", degradation_uri)
        return event_uri

    def add_runtime_provenance(self, payload: dict[str, Any] | None) -> str | None:
        """Record forecast/provider provenance, including STCast when supplied by upstream context."""

        if not payload:
            return None
        provider = payload.get("provider") or payload.get("forecast_model") or payload.get("model") or payload.get("primary_provider")
        if not provider:
            return None
        uri = f"{self.namespace}forecast_source_{self._slug(provider)}_{self._digest(payload)}"
        self.add(uri, "rdf:type", self.namespace + "ForecastSource")
        self.add(uri, "rdfs:label", str(provider))
        for key in ("provider_role", "provider_status", "source_status", "forecast_model", "model_version"):
            if key in payload and payload[key] is not None:
                self.add(uri, self.namespace + self._slug(key), str(payload[key]))
        self.add(uri, self.namespace + "provenancePayload", json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        return uri

    def add_grounding_gap_evidence(
        self,
        risk: HazardRisk,
        *,
        source_metadata: dict[str, Any] | None = None,
        grounding_gap: dict[str, Any] | None = None,
        source_status: str | None = None,
        degradation_metadata: dict[str, Any] | None = None,
        source_uri: str | None = None,
    ) -> list[str]:
        """Persist feature grounding comparisons linked to a risk event.

        Grounding data remains evidence only: this method never changes the risk
        or model-facing feature vector. Nested grounding families become separate
        records while the original family payload remains queryable as JSON.
        """

        metadata = dict(source_metadata or {})
        gaps = grounding_gap if grounding_gap is not None else metadata.get("grounding_gap", {})
        if not isinstance(gaps, dict) or not gaps:
            return []
        event_uri = self._risk_event_uri(risk)
        resolved_sources = metadata.get("resolved_sources", {})
        if not isinstance(resolved_sources, dict):
            resolved_sources = {}
        status = source_status or metadata.get("source_status") or risk.source_status
        degradation = degradation_metadata if degradation_metadata is not None else metadata.get("degradation_metadata", {})
        if not isinstance(degradation, dict):
            degradation = {}
        families = {key: value for key, value in gaps.items() if isinstance(value, dict)}
        if not families:
            families = {"runtime": gaps}
        records: list[str] = []
        for family, comparison in families.items():
            payload = {
                "indicator_family": family,
                "comparison": comparison,
                "source_metadata": metadata,
                "source_status": status,
                "degradation_metadata": degradation,
            }
            record_uri = f"{event_uri}/grounding/{self._slug(family)}_{self._digest(payload)}"
            self.add(record_uri, "rdf:type", self.namespace + "GroundingGap")
            self.add(record_uri, "rdfs:label", str(family))
            self.add(record_uri, self.namespace + "groundingPayload", json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
            self.add(event_uri, self.namespace + "hasGroundingGap", record_uri)
            if source_uri:
                self.add(record_uri, self.namespace + "derivedFromSource", source_uri)
            if status:
                self.add(record_uri, self.namespace + "sourceStatus", str(status))
            if degradation:
                self.add(record_uri, self.namespace + "degradationPayload", json.dumps(degradation, ensure_ascii=False, sort_keys=True, default=str))

            source_pair = comparison.get("source_pair") or comparison.get("sourcePair")
            if isinstance(source_pair, (list, tuple)):
                for source_id in source_pair:
                    self.add(record_uri, self.namespace + "sourcePair", str(source_id))
            elif source_pair:
                self.add(record_uri, self.namespace + "sourcePair", str(source_pair))
            for key, predicate in (
                ("comparison_timestamp", "comparisonTime"),
                ("comparison_time", "comparisonTime"),
                ("timestamp", "comparisonTime"),
                ("units", "units"),
                ("absolute_difference", "absoluteDelta"),
                ("absolute_delta", "absoluteDelta"),
                ("delta", "delta"),
                ("status", "comparisonStatus"),
            ):
                if comparison.get(key) is not None:
                    self.add(record_uri, self.namespace + predicate, str(comparison[key]))
            for source_family, source_info in resolved_sources.items():
                if isinstance(source_info, dict) and source_info.get("dataset_id"):
                    self.add(record_uri, self.namespace + "resolvedSource", str(source_info["dataset_id"]))
            records.append(record_uri)
        return records

    def semantic_evidence_for_risk(self, risk: HazardRisk) -> dict[str, Any]:
        """Return deterministic agent-facing semantic evidence for one risk."""

        impacts = self.query_hazard_impacts(risk.hazard_type)
        return {
            "event_uri": self._risk_event_uri(risk),
            "hazard_type": risk.hazard_type,
            "disaster_type_uri": impacts["disaster_type_uri"],
            "risk_level": risk.risk_level.value,
            "risk_probability": round(float(risk.risk_probability), 6),
            "exposures": impacts["exposures"],
            "actions": impacts["actions"],
            "departments": impacts["departments"],
            "indicator_evidence": dict(risk.indicator_evidence),
            "contributing_factors": list(risk.contributing_factors),
            "model": {
                "family": risk.model_family,
                "name": risk.model_name,
                "version": risk.model_version,
                "inference_mode": risk.inference_mode,
                "source_status": risk.source_status,
            },
            "shap_top_features": self._top_shap_features(risk),
            "degradation_metadata": dict(risk.degradation_metadata),
        }

    def query_hazard_impacts(self, hazard_type: str) -> dict[str, list[str]]:
        """Return affected exposure objects and response actions for a hazard."""

        subject = self.namespace + hazard_type
        exposures = [t.object.removeprefix(self.namespace) for t in self.triples if t.subject == subject and t.predicate == self.namespace + "affectsExposure"]
        actions = [t.object.removeprefix(self.namespace) for t in self.triples if t.subject == subject and t.predicate == self.namespace + "requiresAction"]
        departments = [t.object.removeprefix(self.namespace) for t in self.triples if t.subject == subject and t.predicate == self.namespace + "servedByDepartment"]
        return {
            "hazard_type": [hazard_type],
            "exposures": exposures,
            "actions": actions,
            "departments": departments,
            "disaster_type_uri": self.sakuna_adapter.disaster_type_uri(hazard_type),
        }

    def query_response_actions_by_hazard(self, hazard_type: str) -> dict[str, Any]:
        """Return configured response actions for one hazard type."""

        impacts = self.query_hazard_impacts(hazard_type)
        return {"query_type": "response_actions_by_hazard", "hazard_type": hazard_type, "actions": impacts["actions"]}

    def query_warning_products_by_department(self, department: str) -> dict[str, Any]:
        """Return warning products for hazards served by a department."""

        products = []
        hazards = []
        for hazard, info in self.rules.items():
            if department in info.get("departments", []):
                hazards.append(hazard)
                products.extend(info.get("warning_products", []))
        return {"query_type": "warning_products_by_department", "department": department, "hazards": hazards, "warning_products": sorted(set(products))}

    def query_hazards_by_exposure(self, exposure: str) -> dict[str, Any]:
        """Return hazards affecting an exposure object."""

        hazards = [hazard for hazard, info in self.rules.items() if exposure in info.get("exposures", [])]
        return {"query_type": "hazards_by_exposure", "exposure": exposure, "hazards": hazards}

    def query(self, query_type: str, value: str) -> dict[str, Any]:
        """Stable query facade for API callers."""

        if query_type == "response_actions_by_hazard":
            return self.query_response_actions_by_hazard(value)
        if query_type == "warning_products_by_department":
            return self.query_warning_products_by_department(value)
        if query_type == "hazards_by_exposure":
            return self.query_hazards_by_exposure(value)
        if query_type == "hazard_impacts":
            return self.query_hazard_impacts(value)
        raise ValueError(f"Unsupported query_type: {query_type}")

    def serialize(self, format: str = "ttl") -> str:
        """Serialize graph as ttl, rdfxml, or jsonld."""

        normalized = {"ttl": "turtle", "rdfxml": "xml", "jsonld": "json-ld"}.get(format, format)
        if self._rdflib_graph is not None:
            data = self._rdflib_graph.serialize(format=normalized)
            return data.decode("utf-8") if isinstance(data, bytes) else data
        if normalized != "turtle":
            raise RuntimeError("rdflib is required for rdfxml/jsonld serialization")
        lines = [
            f"@prefix mazu: <{self.namespace}> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        ]
        for triple in self.triples:
            s = triple.subject.replace(self.namespace, "mazu:")
            p = triple.predicate.replace(self.namespace, "mazu:")
            o = triple.object.replace(self.namespace, "mazu:") if triple.object.startswith(self.namespace) else json.dumps(triple.object, ensure_ascii=False)
            lines.append(f"{s} {p} {o} .")
        return "\n".join(lines) + "\n"

    def export_instances(self, format: str = "ttl") -> str:
        """Serialize only MAZU/Sakuna instance triples, excluding loaded ontology triples."""

        normalized = {"ttl": "turtle", "rdfxml": "xml", "jsonld": "json-ld"}.get(format, format)
        if self._rdflib_graph is None:
            if normalized != "turtle":
                raise RuntimeError("rdflib is required for rdfxml/jsonld serialization")
            return self.to_ttl()
        from rdflib import Graph, Literal, Namespace, URIRef

        graph = Graph()
        graph.bind("mazu", Namespace(self.namespace))
        graph.bind("sakuna", Namespace(SAKUNA))
        graph.bind("baw", Namespace(BAWARE))
        graph.bind("geo", Namespace(self.ontology.geosparql_uri))
        for triple in self.triples:
            s = URIRef(triple.subject) if triple.subject.startswith("http") else URIRef(self.namespace + triple.subject)
            p = self._rdf_type if triple.predicate == "rdf:type" else self._rdfs_label if triple.predicate == "rdfs:label" else URIRef(triple.predicate)
            o = URIRef(triple.object) if triple.object.startswith("http") else Literal(triple.object)
            graph.add((s, p, o))
        data = graph.serialize(format=normalized)
        return data.decode("utf-8") if isinstance(data, bytes) else data

    def push_to_graphdb(self, endpoint: str | None = None, format: str = "ttl") -> dict[str, Any]:
        """POST MAZU instance triples to a configured GraphDB statements endpoint."""

        target = (endpoint or self.sakuna_adapter.graphdb_endpoint or "").strip()
        if not target:
            return {
                "status": "skipped",
                "persisted": False,
                "reason": "graphdb_endpoint_not_configured",
                "triple_count": len(self.triples),
            }
        payload = self.export_instances(format=format).encode("utf-8")
        request = urllib.request.Request(
            target,
            data=payload,
            headers={"Content-Type": "text/turtle", "Accept": "application/json, text/plain, */*"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
                return {
                    "status": "persisted",
                    "persisted": True,
                    "http_status": response.status,
                    "triple_count": len(self.triples),
                    "endpoint": target,
                    "response_preview": body[:500],
                }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return {
                "status": "error",
                "persisted": False,
                "http_status": exc.code,
                "triple_count": len(self.triples),
                "endpoint": target,
                "error": body[:500],
            }
        except Exception as exc:
            return {
                "status": "error",
                "persisted": False,
                "triple_count": len(self.triples),
                "endpoint": target,
                "error": str(exc),
            }

    def to_ttl(self) -> str:
        """Serialize graph to Turtle."""

        return self.serialize("ttl")

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph summary."""

        return {
            "namespace": self.namespace,
            "backend": self.backend_metadata,
            "triple_count": len(self.triples),
            "triples": [t.to_tuple() for t in self.triples],
        }


class SakunaGraphAdapter:
    """Local SakunaGraPH ontology adapter.

    SakunaGraPH contributes the disaster ontology and RDF/SPARQL graph shape.
    MAZU still injects Saudi-specific hazard response rules and live risk
    evidence as operational instance data.
    """

    def __init__(self, ttl_path: str | Path | None = None, graphdb_endpoint: str | None = None) -> None:
        self.ttl_path = self._resolve_ttl_path(ttl_path)
        self.graphdb_endpoint = (graphdb_endpoint or os.getenv("MAZU_SAKUNA_GRAPHDB_ENDPOINT", "")).strip()
        self._graph = None
        self._metadata: dict[str, Any] = {
            "backend": "sakuna",
            "status": "unavailable",
            "ontology_path": str(self.ttl_path) if self.ttl_path else None,
            "ontology_triple_count": 0,
            "graphdb_endpoint": self.graphdb_endpoint or None,
        }

    @staticmethod
    def _resolve_ttl_path(explicit: str | Path | None = None) -> Path | None:
        if explicit is not None:
            return Path(explicit)
        env_path = os.getenv("MAZU_SAKUNA_TTL")
        if env_path:
            return Path(env_path)
        root = Path(__file__).resolve().parents[3]
        candidate = root / "SakunaGraPH" / "ontology" / "sakunagraph.ttl"
        return candidate if candidate.exists() else None

    def available(self) -> bool:
        """Return whether the local SakunaGraPH ontology file is readable."""

        return self.ttl_path is not None and self.ttl_path.exists()

    def load_graph(self):
        """Load the local SakunaGraPH Turtle ontology into an rdflib graph."""

        if not self.available():
            raise FileNotFoundError("SakunaGraPH ontology not found. Set MAZU_SAKUNA_TTL to sakunagraph.ttl.")
        from rdflib import Graph

        graph = Graph()
        graph.parse(str(self.ttl_path), format="turtle")
        self._graph = graph
        self._metadata = {
            "backend": "sakuna",
            "status": "local_ontology_loaded",
            "ontology_path": str(self.ttl_path),
            "ontology_triple_count": len(graph),
            "graphdb_endpoint": self.graphdb_endpoint or None,
        }
        return graph

    def metadata(self) -> dict[str, Any]:
        """Return adapter runtime metadata."""

        return dict(self._metadata)

    def disaster_type_uri(self, hazard_type: str) -> str | None:
        """Map MAZU hazard names to SakunaGraPH disaster-type concepts when possible."""

        mapping = {
            "flash_flood": SAKUNA + "FlashFlood",
            "extreme_heat": SAKUNA + "ExtremeTemperature",
            "dry_heat_agriculture": SAKUNA + "Drought",
            "dust_potential": SAKUNA + "SandStorm",
            "coastal_humid_heat": SAKUNA + "ExtremeTemperature",
        }
        return mapping.get(hazard_type)

    def push(self, graph: HazardKnowledgeGraph) -> dict[str, Any]:
        """Persist when GraphDB is configured, otherwise summarize readiness."""

        if self.graphdb_endpoint:
            return graph.push_to_graphdb(self.graphdb_endpoint)
        return {
            "status": "ready" if self.available() else "unavailable",
            "persisted": False,
            "reason": "graphdb_endpoint_not_configured",
            "triple_count": len(graph.triples),
            "target": "SakunaGraPH",
            "graphdb_endpoint": None,
        }


class GeoSparqlQueryService:
    """Placeholder for future GeoSPARQL spatial reasoning."""

    def query_bbox(self, bbox: tuple[float, float, float, float], feature_type: str | None = None) -> dict[str, Any]:
        """Return a placeholder spatial query response."""

        return {"status": "placeholder", "bbox": bbox, "feature_type": feature_type, "features": []}


class ShaclValidationService:
    """Placeholder for future SHACL validation."""

    def validate(self, graph: HazardKnowledgeGraph) -> dict[str, Any]:
        """Return a minimal validation response."""

        required_classes = {"HazardScenario", "ExposureObject", "ResponseAction", "ServiceDepartment"}
        present = {t.object.removeprefix(graph.namespace) for t in graph.triples if t.predicate == "rdf:type"}
        missing = sorted(required_classes - present)
        return {
            "conforms": not missing,
            "status": "placeholder",
            "triple_count": len(graph.triples),
            "results": [{"severity": "warning", "message": f"Missing class seed: {item}"} for item in missing],
        }
