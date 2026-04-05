"""Typed HTTP boundary models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    type: str
    param: str | None
    code: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorBody


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: str = "list"
    data: list[ModelCard]


class ProviderHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    authenticated: bool
    method: str | None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    providers: list[ProviderHealth]
