"""RDF/OWL-style knowledge graph minimum closed loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mazu_saudi.schemas import HazardRisk

MAZU = "https://mazu.example.org/saudi#"


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


@dataclass(frozen=True)
class Triple:
    """Simple RDF-like triple."""

    subject: str
    predicate: str
    object: str

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


class HazardKnowledgeGraph:
    """Minimal RDF triple generator and query facade."""

    def __init__(self, ontology: OntologyConfig | None = None, rules: dict[str, Any] | None = None) -> None:
        self.ontology = ontology or OntologyConfig()
        self.rules = rules or DEFAULT_HAZARD_KNOWLEDGE
        self.namespace = self.ontology.base_uri
        self.triples: list[Triple] = []
        self._rdflib_graph = None
        try:
            from rdflib import Graph, Namespace, URIRef

            self._rdflib_graph = Graph()
            self._namespace = Namespace(self.namespace)
            self._rdflib_graph.bind("mazu", self._namespace)
            self._rdflib_graph.bind("geo", Namespace(self.ontology.geosparql_uri))
            self._rdflib_graph.bind("sh", Namespace(self.ontology.shacl_uri))
            self._rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            self._rdfs_label = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
        except Exception:
            self._namespace = None
        self._seed_ontology()

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

    def query_hazard_impacts(self, hazard_type: str) -> dict[str, list[str]]:
        """Return affected exposure objects and response actions for a hazard."""

        subject = self.namespace + hazard_type
        exposures = [t.object.removeprefix(self.namespace) for t in self.triples if t.subject == subject and t.predicate == self.namespace + "affectsExposure"]
        actions = [t.object.removeprefix(self.namespace) for t in self.triples if t.subject == subject and t.predicate == self.namespace + "requiresAction"]
        departments = [t.object.removeprefix(self.namespace) for t in self.triples if t.subject == subject and t.predicate == self.namespace + "servedByDepartment"]
        return {"hazard_type": [hazard_type], "exposures": exposures, "actions": actions, "departments": departments}

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

    def to_ttl(self) -> str:
        """Serialize graph to Turtle."""

        return self.serialize("ttl")

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph summary."""

        return {"namespace": self.namespace, "triple_count": len(self.triples), "triples": [t.to_tuple() for t in self.triples]}


class SakunaGraphAdapter:
    """Placeholder adapter for future SakunaGraPH integration."""

    def push(self, graph: HazardKnowledgeGraph) -> dict[str, Any]:
        """Return a placeholder push result."""

        return {"status": "placeholder", "triple_count": len(graph.triples), "target": "SakunaGraPH"}


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
