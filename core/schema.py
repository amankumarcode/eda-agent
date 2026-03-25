from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class OutputType(str, Enum):
    REPORT    = "report"
    EMAIL     = "email"
    JSON      = "json"
    DASHBOARD = "dashboard"


class EDARequest(BaseModel):
    goal:           str
    session_id:     str
    output_formats: list[OutputType] = [OutputType.REPORT, OutputType.JSON]
    metadata:       dict[str, Any]   = Field(default_factory=dict)


class AnalysisPlan(BaseModel):
    session_id:     str
    goal:           str
    steps:          list[str]
    agents:         list[str]
    parallel:       list[list[str]]
    output_formats: list[OutputType]


class AgentResult(BaseModel):
    agent_name:  str
    success:     bool
    findings:    dict[str, Any]
    confidence:  float = Field(ge=0.0, le=1.0)
    warnings:    list[str]  = Field(default_factory=list)
    rerun_hint:  Optional[str] = None


class ProfilerFindings(BaseModel):
    schema_info:         dict[str, Any]
    null_report:         dict[str, Any]   = Field(default_factory=dict)
    distributions:       dict[str, Any]   = Field(default_factory=dict)
    notable_columns:     list[str]        = Field(default_factory=list)
    categorical_summary: dict[str, Any]   = Field(default_factory=dict)
    data_quality:        dict[str, Any]   = Field(default_factory=dict)


class StatFindings(BaseModel):
    correlations:     dict[str, Any]      = Field(default_factory=dict)
    outliers:         dict[str, Any]      = Field(default_factory=dict)
    normality:        dict[str, Any]      = Field(default_factory=dict)
    skewness:         dict[str, Any]      = Field(default_factory=dict)
    feature_ranking:  list[dict[str, Any]] = Field(default_factory=list)
    notable_findings: list[str]           = Field(default_factory=list)


class VizFindings(BaseModel):
    charts:                    list[dict[str, Any]]
    chart_descriptions:        list[str]  = Field(default_factory=list)
    recommended_primary_chart: str = ""


class CriticOutput(BaseModel):
    scored_results:  list[AgentResult]
    rerun_agent:     Optional[str]   = None
    rerun_reason:    Optional[str]   = None
    overall_quality: float           = Field(ge=0.0, le=1.0)


class NarratorResult(BaseModel):
    narrative:               str
    key_insights:            list[str] = Field(default_factory=list)
    caveats:                 list[str] = Field(default_factory=list)
    recommended_next_steps:  list[str] = Field(default_factory=list)


class REPLRequest(BaseModel):
    code:        str
    description: str


class REPLOutput(BaseModel):
    success:    bool
    result:     dict[str, Any]
    stdout_raw: str
    error:      Optional[str] = None


class Verdict(str, Enum):
    STRONG   = "strong"
    ADEQUATE = "adequate"
    WEAK     = "weak"


class EvaluationResult(BaseModel):
    goal_coverage:      float           = Field(ge=0.0, le=1.0)
    insight_quality:    float           = Field(ge=0.0, le=1.0)
    evidence_quality:   float           = Field(ge=0.0, le=1.0)
    overall_score:      float           = Field(ge=0.0, le=1.0)
    strengths:          list[str]       = Field(default_factory=list)
    gaps:               list[str]       = Field(default_factory=list)
    verdict:            Verdict
    retry_instructions: Optional[str]  = None
