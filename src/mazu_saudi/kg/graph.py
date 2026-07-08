"""RDF/OWL-style knowledge graph minimum closed loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mazu_saudi.schemas import HazardRisk

MAZU = "https://mazu.example.org/saudi#"

DEFAULT_HAZARD_KNOWLEDGE = {
    "flash_flood": {
        "label": "山洪暴雨风险",
        "exposures": ["wadi_communities", "urban_drainage", "roads"],
        "actions": ["activate_flood_watch", "inspect_drainage", "avoid_wadi_crossing"],
        "departments": ["meteorology", "emergency", "transport"],
    },
    "extreme_heat": {
        "label": "极端高温与热健康风险",
        "exposures": ["outdoor_workers", "elderly_population", "power_grid"],
        "actions": ["open_cooling_centers", "adjust_work_hours", "issue_heat_health_advice"],
        "departments": ["public_health", "emergency", "meteorology"],
    },
    "dry_heat_agriculture": {
        "label": "干热农业胁迫风险",
        "exposures": ["date_palm_farms", "greenhouses", "irrigation_systems"],
        "actions": ["increase_irrigation_monitoring", "shade_sensitive_crops", "monitor_vpd"],
        "departments": ["agriculture", "meteorology"],
    },
    "dust_potential": {
        "label": "强风沙尘起沙潜势",
        "exposures": ["highways", "airports", "construction_sites"],
        "actions": ["issue_visibility_warning", "secure_loose_material", "prepare_traffic_controls"],
        "departments": ["transport", "emergency", "meteorology"],
    },
    "coastal_humid_heat": {
        "label": "沿海湿热与水汽输送风险",
        "exposures": ["ports", "coastal_workers", "desalination_plants"],
        "actions": ["monitor_port_operations", "reduce_heat_exposure", "track_moisture_transport"],
        "departments": ["port", "public_health", "meteorology"],
    },
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
    """Minimal RDF triple generator and query facade."""

    def __init__(self) -> None:
        self.triples: list[Triple] = []
        self._rdflib_graph = None
        try:
            from rdflib import Graph, Namespace

            self._rdflib_graph = Graph()
            self._namespace = Namespace(MAZU)
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
        for hazard, info in DEFAULT_HAZARD_KNOWLEDGE.items():
            subject = MAZU + hazard
            self.add(subject, "rdf:type", MAZU + "HazardScenario")
            self.add(subject, "rdfs:label", info["label"])
            for exposure in info["exposures"]:
                self.add(subject, MAZU + "affectsExposure", MAZU + exposure)
                self.add(MAZU + exposure, "rdf:type", MAZU + "ExposureObject")
            for action in info["actions"]:
                self.add(subject, MAZU + "requiresAction", MAZU + action)
                self.add(MAZU + action, "rdf:type", MAZU + "ResponseAction")
            for department in info["departments"]:
                self.add(subject, MAZU + "servedByDepartment", MAZU + department)
                self.add(MAZU + department, "rdf:type", MAZU + "ServiceDepartment")

    def add(self, subject: str, predicate: str, obj: str) -> None:
        """Add a triple to the internal graph."""

        triple = Triple(subject, predicate, obj)
        if triple not in self.triples:
            self.triples.append(triple)
        if self._rdflib_graph is not None:
            from rdflib import Literal, URIRef

            s = URIRef(subject) if subject.startswith("http") else URIRef(MAZU + subject)
            p = URIRef(predicate) if predicate.startswith("http") else URIRef(MAZU + predicate.replace(":", "_"))
            o = URIRef(obj) if obj.startswith("http") else Literal(obj)
            self._rdflib_graph.add((s, p, o))

    def add_risk_evidence(self, risk: HazardRisk) -> None:
        """Generate triples connecting a risk output to indicators and level."""

        risk_id = f"{MAZU}risk_{risk.hazard_type}_{risk.grid.id if risk.grid else 'unknown'}"
        hazard_uri = MAZU + risk.hazard_type
        self.add(risk_id, "rdf:type", MAZU + "WarningProduct")
        self.add(risk_id, MAZU + "warnsAbout", hazard_uri)
        self.add(risk_id, MAZU + "hasRiskLevel", risk.risk_level.value)
        for factor in risk.contributing_factors:
            indicator_id = MAZU + "indicator_" + str(abs(hash((risk.hazard_type, factor))))
            self.add(indicator_id, "rdf:type", MAZU + "MeteorologicalIndicator")
            self.add(risk_id, MAZU + "hasContributingFactor", indicator_id)
            self.add(indicator_id, "rdfs:label", factor)

    def query_hazard_impacts(self, hazard_type: str) -> dict[str, list[str]]:
        """Return affected exposure objects and response actions for a hazard."""

        subject = MAZU + hazard_type
        exposures = [t.object.removeprefix(MAZU) for t in self.triples if t.subject == subject and t.predicate == MAZU + "affectsExposure"]
        actions = [t.object.removeprefix(MAZU) for t in self.triples if t.subject == subject and t.predicate == MAZU + "requiresAction"]
        departments = [t.object.removeprefix(MAZU) for t in self.triples if t.subject == subject and t.predicate == MAZU + "servedByDepartment"]
        return {"hazard_type": [hazard_type], "exposures": exposures, "actions": actions, "departments": departments}

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph summary."""

        return {"namespace": MAZU, "triple_count": len(self.triples), "triples": [t.to_tuple() for t in self.triples]}


class SakunaGraphAdapter:
    """Placeholder adapter for future SakunaGraPH integration."""

    def push(self, graph: HazardKnowledgeGraph) -> dict[str, Any]:
        """Return a placeholder push result."""

        return {"status": "placeholder", "triple_count": len(graph.triples), "target": "SakunaGraPH"}


class GeoSparqlQueryService:
    """Placeholder for future GeoSPARQL spatial reasoning."""

    def query_bbox(self, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
        """Return a placeholder spatial query response."""

        return {"status": "placeholder", "bbox": bbox, "features": []}


class ShaclValidationService:
    """Placeholder for future SHACL validation."""

    def validate(self, graph: HazardKnowledgeGraph) -> dict[str, Any]:
        """Return a minimal validation response."""

        return {"conforms": True, "status": "placeholder", "triple_count": len(graph.triples)}
