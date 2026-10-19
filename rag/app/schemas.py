"""Pydantic models for every JSON reply an LLM stage returns. Parsing through
these replaces hand-written .get()/isinstance checks: a reply that doesn't fit
raises ValidationError, and each caller falls back the same way it always has."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QueryType = Literal["simple", "research", "conversational", "metadata", "out_of_scope"]


class MetadataFilter(BaseModel):
    column: str = ""
    value: bool | str = True


class RouterReply(BaseModel):
    """router.txt"""
    rewritten_query: str = ""
    keywords: list[str] | None = None
    type: QueryType = "research"
    needs_retrieval: bool = True
    metadata_filter: MetadataFilter | None = None

    @field_validator("type", mode="before")
    @classmethod
    def unknown_type_is_research(cls, v):
        # An unexpected label used to fall through to the research path; keep that.
        return v if v in QueryType.__args__ else "research"


class Route(BaseModel):
    """What the pipeline acts on after routing."""
    query_type: QueryType
    rewritten_query: str
    search_terms: list[str]
    needs_retrieval: bool = True
    metadata_filter: MetadataFilter | None = None


class GroundingVerdict(BaseModel):
    """grounding-judge.txt"""
    grounded: bool = True
    reason: str = ""


class UnsupportedCitation(BaseModel):
    """One entry of citation-verify.txt's "unsupported" list."""
    unit: int
    source: int
    correct_source: int | None = None


class CitationAudit(BaseModel):
    # Items are validated one by one in verification.verify_citations, so a
    # malformed entry is skipped instead of discarding the whole audit.
    unsupported: list = Field(default_factory=list)


class ExpansionReply(BaseModel):
    """expansion.txt"""
    queries: list[str] = Field(default_factory=list)


# reranker.txt. Also sent to the model as a strict JSON schema, so no
# docstring: pydantic would ship it to the model as the schema description.
class RerankReply(BaseModel):
    model_config = ConfigDict(extra="forbid", title="rerank")
    indices: list[int]
