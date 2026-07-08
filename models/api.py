"""Shared API request/response models."""

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class WorkspaceRequest(BaseModel):
    """Request payload for creating a workspace."""

    name: str


class ResolveRequest(BaseModel):
    """Request payload for resolving duplicate files."""

    file_id: str
    action: Literal["delete", "keep"]


class WorkflowRunRequest(BaseModel):
    """Request payload for running the workflow against an index."""

    index_path: str = Field(
        ...,
        validation_alias=AliasChoices("index_path", "vector_db"),
    )


class WorkflowHumanResponseRequest(BaseModel):
    """Request payload for responding to a workflow HITL checkpoint."""

    run_id: str
    request_id: str
    action: Literal["approve", "feedback", "exit"]
    text: str = ""

class ChatbotRequest(BaseModel):
    """Request payload for asking a question against the session vector index."""

    question: str
