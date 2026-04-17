"""Typed configuration schema using Pydantic v2."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AuthSettings(BaseModel):
    username: str = ""
    password: str = ""
    token: str = ""
    api_key: str = ""

    model_config = {"extra": "allow"}


class TLSSettings(BaseModel):
    verify: bool = True
    ca_cert: Optional[str] = None
    ca_cert_base64: Optional[str] = None


class ArangoSettings(BaseModel):
    mode: Literal["local", "cloud"] = "local"
    url: str = "http://localhost:8529"
    port: int = 8529
    database: str = "memwright"
    auth: AuthSettings = Field(default_factory=AuthSettings)
    tls: TLSSettings = Field(default_factory=TLSSettings)
    docker: bool = False


class PostgresSettings(BaseModel):
    mode: Literal["local", "cloud"] = "local"
    url: str = "postgresql://localhost:5432"
    port: int = 5432
    database: str = "memwright"
    auth: AuthSettings = Field(default_factory=AuthSettings)
    tls: TLSSettings = Field(default_factory=TLSSettings)


class MemwrightSettings(BaseModel):
    """Top-level memwright configuration."""

    backends: List[str] = Field(default_factory=lambda: ["sqlite", "chroma", "networkx"])
    default_token_budget: int = Field(16000, gt=0)
    min_results: int = Field(3, ge=0)
    enable_mmr: bool = True
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    fusion_mode: Literal["rrf", "graph_blend"] = "rrf"
    confidence_gate: float = Field(0.0, ge=0.0, le=1.0)
    confidence_decay_rate: float = Field(0.001, ge=0.0)
    confidence_boost_rate: float = Field(0.03, ge=0.0)

    arangodb: ArangoSettings = Field(default_factory=ArangoSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)

    profiles: Dict[str, Dict] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @field_validator("backends")
    @classmethod
    def _backends_non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("backends list cannot be empty")
        return v
