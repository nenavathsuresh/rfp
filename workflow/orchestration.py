"""Orchestration logic for running Stage 1 Opportunity Analysis workflow with RAG tools and HITL."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_framework import AgentResponseUpdate, WorkflowBuilder
from agent_framework_orchestrations import AgentRequestInfoResponse
from agent_framework_orchestrations._orchestration_request_info import AgentApprovalExecutor

from config.settings import get_settings
from agents.requirements_extractor_agent import create_requirements_extractor_agent,create_requirements_extractor_agent_markdown
from agents.compliance_agent import create_compliance_agent
from agents.evidence_agent import create_evidence_agent
from observability import CostTracker



class WorkflowExitRequested(RuntimeError):
    """Raised when the reviewer asks to stop the workflow."""


async def _emit(emit: Any, payload: dict[str, Any]) -> None:
    """Send a structured workflow event to an optional UI stream."""

    if emit is not None:
        await emit(payload)


def _new_agent_run_output_path(agent_slug: str, extension: str = "json") -> Path:
    """Create a per-run path for persisted agent output."""

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    normalized_extension = extension.lstrip(".")
    return runs_dir / f"{agent_slug}_{timestamp}.{normalized_extension}"


def _new_workflow_log_path() -> Path:
    """Create a per-run JSONL log path for agent and human communication."""

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"workflow_communication_{timestamp}.jsonl"


def _write_log(log_path: Path, event_type: str, **payload: Any) -> None:
    """Append one structured communication event to the run log."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _save_agent_output_json(agent_name: str, output_text: str, log_path: Path) -> Path:
    """Persist requirements extractor output as normalized JSON payload."""

    output_path = _new_agent_run_output_path("requirements_extractor", "json")
    payload = _normalize_requirements_payload(output_text)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=4), encoding="utf-8")
    return output_path

def _save_agent_output_markdown(agent_name: str, output_text: str, log_path: Path) -> Path:
    """Persist requirements extractor output as Markdown."""

    output_path = _new_agent_run_output_path("requirements_extractor", "md")
    output_path.write_text((output_text or "").strip() + "\n", encoding="utf-8")
    return output_path


def _normalize_requirements_payload(output_text: str) -> dict[str, Any]:
    """Normalize model output into {"requirements": [...]} schema."""

    cleaned = (output_text or "").strip()
    if not cleaned:
        return {"requirements": []}

    parsed = _try_parse_json_payload(cleaned)
    if parsed is None:
        return {"requirements": []}

    if isinstance(parsed, dict):
        if "requirements" in parsed:
            requirements = parsed.get("requirements")
            if isinstance(requirements, list):
                return {"requirements": requirements}
            if isinstance(requirements, dict):
                return {"requirements": [requirements]}
            return {"requirements": []}

        # Fallback: single requirement object returned directly.
        if any(
            key in parsed
            for key in (
                "Requirement_ID",
                "Requirement_type",
                "Requirement_name",
                "section",
                "page_source",
                "confidence",
            )
        ):
            return {"requirements": [parsed]}
        return {"requirements": []}

    if isinstance(parsed, list):
        return {"requirements": parsed}

    return {"requirements": []}


def _try_parse_json_payload(text: str) -> Any | None:
    """Parse JSON from plain text, fenced blocks, or extracted braces."""

    candidates = [text]

    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            candidate = chunk.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            candidates.append(candidate)

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        candidates.append(text[array_start : array_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_response_text(response: Any) -> str:
    """Extract readable text from Agent Framework response objects."""

    nested_response = getattr(response, "agent_response", None)
    if nested_response is not None and nested_response is not response:
        nested_text = _extract_response_text(nested_response)
        if nested_text:
            return nested_text

    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text

    messages = getattr(response, "messages", None)
    if messages:
        message_text = [
            getattr(message, "text", "")
            for message in messages
            if getattr(message, "text", "")
        ]
        return "\n".join(message_text)

    return ""


def _with_human_approval(agent: Any, *, terminal: bool = False) -> Any:
    """Wrap an agent so Microsoft Agent Framework emits request_info approval events."""

    return AgentApprovalExecutor(agent, allow_direct_output=terminal)


def _compose_agent_input(user_prompt: str, context_text: str) -> str:
    """Compose the actual runtime input sent to each agent run."""

    prompt = (user_prompt or "").strip()
    context = (context_text or "").strip()
    if prompt and context:
        return f"{prompt}\n\nContext:\n{context}"
    return prompt or context


def _request_human_response(
    agent_name: str,
    log_path: Path,
    agent_text: str = "",
    agent_user_prompt: str = "",
) -> AgentRequestInfoResponse:
    """Prompt a human reviewer for workflow approval or evidence Q/A input."""

    _write_log(
        log_path,
        "human_checkpoint",
        agent_name=agent_name,
        agent_text=agent_text,
    )

    is_evidence_agent = "evidence" in agent_name.lower()
    if is_evidence_agent:
        user_input = input("\nEvidence query (type 'exit' to finish): ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            _write_log(
                log_path,
                "evidence_review_exit",
                agent_name=agent_name,
                human_input=user_input,
            )
            return AgentRequestInfoResponse.approve()
        if not user_input:
            print("Please enter an evidence query, or type 'exit' to finish.")
            return AgentRequestInfoResponse.from_strings([
                "Ask the reviewer for a specific evidence query before performing validation."
            ])

        combined_feedback = _compose_agent_input(agent_user_prompt, user_input)
        print(f"[Evidence Query Captured]: {user_input}")
        _write_log(
            log_path,
            "evidence_review_query",
            agent_name=agent_name,
            human_input=user_input,
        )
        return AgentRequestInfoResponse.from_strings([combined_feedback])

    user_input = input(
        f"\n--- [HITL Checkpoint] Review output from {agent_name} ---\n"
        "Press [Enter] to approve, type feedback to revise, or type 'exit' to stop: "
    ).strip()

    if user_input.lower() in {"exit", "quit", "q"}:
        _write_log(
            log_path,
            "human_exit",
            agent_name=agent_name,
            human_input=user_input,
        )
        raise WorkflowExitRequested(f"Workflow stopped by reviewer at {agent_name}.")

    if user_input:
        combined_feedback = _compose_agent_input(agent_user_prompt, user_input)
        print(f"[HITL Feedback Captured for {agent_name}]: {user_input}")
        _write_log(
            log_path,
            "human_feedback",
            agent_name=agent_name,
            human_input=user_input,
        )
        return AgentRequestInfoResponse.from_strings([combined_feedback])

    print(f"[HITL Approved]: {agent_name}")
    _write_log(
        log_path,
        "human_approval",
        agent_name=agent_name,
        human_input="",
    )
    return AgentRequestInfoResponse.approve()

async def run_opportunity_analysis(
    search_tool: Callable[..., Any],
    md_path: str | None = None,
    #rfp_text_summary: str,
    usage_tracker: CostTracker | None = None,
    provider_hint: str | None = None,
    model_hint: str | None = None,
    emit: Any | None = None,
    human_response_queue: Any | None = None,
) -> dict[str, str]:
    """Execute the Stage 1 Opportunity Analysis workflow using Microsoft Agent Framework Workflows.
    
    Includes:
    - Requirements extraction followed by extracted-output Q/A.
    - Graph layout matching the Stage 1 branching architecture.
    - Human-in-the-loop checkpoints before critical transition steps.
    """
    #settings = get_settings()
    #print(settings)
    settings = get_settings()
    
    # 1. Build specialized agents, passing the RAG search tool to EACH agent
    print("Building Stage 1 workflow agents...")
    await _emit(emit, {"type": "log", "message": "Building Stage 1 workflow agents..."})
    # Alternative RAG-backed extractor path can use create_requirements_extractor_agent(search_tool=search_tool).
    # requirements_agent_name = str(getattr(requirements_agent_base, "name", "") or "")
    # requirements_prompt = str(getattr(requirements_agent_base, "total_prompt", "") or "")
    # requirements_extractor = _with_human_approval(requirements_agent_base)
    
    
    requirements_agent_base = create_requirements_extractor_agent_markdown()
    requirements_agent_name = str(getattr(requirements_agent_base, "name", "") or "")
    requirements_prompt = str(getattr(requirements_agent_base, "user_prompt", "") or "")
    
    # md_dir = Path(md_path)
    # md_file = next(md_dir.glob("*.md"))
    # markdown_data = md_file.read_text(encoding="utf-8")
   
    md_dir = Path(md_path)
    markdown_data = ""
    for md_file in md_dir.glob("*.md"):
        markdown_data += md_file.read_text(encoding="utf-8") + "\n\n"
    root = Path.cwd()
    TEMPLATE_MARKDOWN = Path(root / "instructions" / "RFP_Generalized_Template.md").read_text(encoding="utf-8")
    
    requirements_prompt = (
    requirements_prompt
    .replace("{{TEMPLATE_MARKDOWN}}", TEMPLATE_MARKDOWN)
    .replace("{{SOURCE_RFP_DOCUMENT_TEXT_OR_FILES}}", markdown_data)
    )

    

    requirements_extractor = _with_human_approval(requirements_agent_base)
    evidence_agent_base = create_evidence_agent()
    evidence_executor = _with_human_approval(evidence_agent_base)
    
    
    agent_user_prompts = {
        str(getattr(requirements_agent_base, "name", "") or ""): requirements_prompt,
        str(getattr(evidence_agent_base, "name", "") or ""): str(getattr(evidence_agent_base, "user_prompt", "") or ""),
    }
    
    
    # 2. Construct the Stage 1 workflow graph.
    builder = WorkflowBuilder(start_executor=requirements_extractor)
    builder.add_edge(requirements_extractor, evidence_executor)
    
    workflow = builder.build()
    
    # 3. Execute the workflow with framework-native Human-In-The-Loop checkpoints
    results = {}
    log_path = _new_workflow_log_path()
    
    print("\nStarting Stage 1 workflow execution...")
    print(f"Communication log: {log_path}")
    await _emit(emit, {"type": "log", "message": "Starting Stage 1 workflow execution..."})
    await _emit(emit, {"type": "log", "message": f"Communication log: {log_path}"})
    _write_log(
        log_path,
        "workflow_start",
        #rfp_text_summary=rfp_text_summary,
        human_approval_enabled=settings.human_approval_enabled,
    )

    #initial_input = _compose_agent_input(requirements_prompt)
    stream = workflow.run(requirements_prompt,stream=True)
    try:
        while True:
            pending_responses: dict[str, AgentRequestInfoResponse] = {}

            async for event in stream:
                if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
                    update = event.data
                    author = update.author_name or "Agent"
                    if author not in results:
                        results[author] = ""
                        print(f"\n[{author}]: ", end="", flush=True)
                        await _emit(emit, {"type": "agent_start", "agent": author})

                    results[author] += update.text
                    print(update.text, end="", flush=True)
                    await _emit(emit, {"type": "agent_delta", "agent": author, "text": update.text})
                    _write_log(
                        log_path,
                        "agent_output_delta",
                        agent_name=author,
                        text=update.text,
                    )
                    if usage_tracker is not None:
                        usage_tracker.try_record_llm_from_response(
                            response=update,
                            agent_name=author,
                            request_id=event.request_id,
                            provider_hint=provider_hint,
                            model_hint=model_hint,
                            metadata={"event_type": event.type, "usage_source": "output_update"},
                        )

                elif event.type == "request_info":
                    request_data = event.data
                    agent_response = getattr(request_data, "agent_response", None)
                    agent_name = getattr(agent_response, "author_name", None) or event.source_executor_id
                    agent_text = _extract_response_text(agent_response)
                    selected_prompt = agent_user_prompts.get(agent_name, "")
                    usage_recorded = False
                    if usage_tracker is not None:
                        # Some providers expose usage on the request_info payload, not only on agent_response.
                        usage_recorded = usage_tracker.try_record_llm_from_response(
                            response=request_data,
                            agent_name=agent_name,
                            request_id=event.request_id,
                            provider_hint=provider_hint,
                            model_hint=model_hint,
                            metadata={"event_type": event.type, "usage_source": "request_info"},
                        )
                        if not usage_recorded:
                            usage_recorded = usage_tracker.try_record_llm_from_response(
                                response=agent_response,
                                agent_name=agent_name,
                                request_id=event.request_id,
                                provider_hint=provider_hint,
                                model_hint=model_hint,
                                metadata={"event_type": event.type, "usage_source": "agent_response"},
                            )
                        if not usage_recorded:
                            estimated_runtime_prompt = _compose_agent_input(selected_prompt, agent_text)
                            estimated_prompt_tokens = usage_tracker.estimate_tokens(estimated_runtime_prompt)
                            estimated_completion_tokens = usage_tracker.estimate_tokens(agent_text)
                            usage_tracker.record_llm_call(
                                agent_name=agent_name,
                                provider=provider_hint or "unknown",
                                model=model_hint or "unknown",
                                request_id=event.request_id,
                                usage={
                                    "prompt_tokens": estimated_prompt_tokens,
                                    "completion_tokens": estimated_completion_tokens,
                                    "total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
                                },
                                metadata={
                                    "event_type": event.type,
                                    "estimated": True,
                                    "reason": "provider usage metadata not available",
                                    "estimated_prompt_source": "agent_user_prompt_plus_context",
                                },
                            )

                    if agent_text:
                        results[agent_name] = agent_text
                        print(f"\n[{agent_name}]:\n{agent_text}", flush=True)
                        _write_log(
                            log_path,
                            "agent_checkpoint_output",
                            agent_name=agent_name,
                            text=agent_text,
                        )
                        await _emit(emit, {
                            "type": "agent_checkpoint",
                            "agent": agent_name,
                            "text": agent_text,
                        })

                    if settings.human_approval_enabled:
                        await _emit(emit, {
                            "type": "hitl_request",
                            "request_id": event.request_id,
                            "agent": agent_name,
                            "text": agent_text,
                            "is_evidence_agent": "evidence" in agent_name.lower(),
                        })
                        if human_response_queue is None:
                            pending_responses[event.request_id] = _request_human_response(
                                agent_name,
                                log_path,
                                agent_text,
                                selected_prompt,
                            )
                        else:
                            response_payload = await human_response_queue.get()
                            while response_payload.get("request_id") != event.request_id:
                                response_payload = await human_response_queue.get()
                            action = response_payload.get("action", "feedback")
                            human_text = str(response_payload.get("text", "")).strip()
                            if action == "exit":
                                _write_log(log_path, "human_exit", agent_name=agent_name, human_input=human_text)
                                raise WorkflowExitRequested(f"Workflow stopped by reviewer at {agent_name}.")
                            if action == "approve":
                                _write_log(log_path, "human_approval", agent_name=agent_name, human_input="")
                                pending_responses[event.request_id] = AgentRequestInfoResponse.approve()
                            else:
                                combined_feedback = _compose_agent_input(selected_prompt, human_text)
                                _write_log(log_path, "human_feedback", agent_name=agent_name, human_input=human_text)
                                pending_responses[event.request_id] = AgentRequestInfoResponse.from_strings([combined_feedback])
                            await _emit(emit, {"type": "hitl_response_received", "request_id": event.request_id})
                    else:
                        _write_log(
                            log_path,
                            "human_auto_approval",
                            agent_name=agent_name,
                        )
                        pending_responses[event.request_id] = AgentRequestInfoResponse.approve()

            if not pending_responses:
                break

            stream = workflow.run(stream=True, responses=pending_responses)
    except WorkflowExitRequested as exc:
        print(f"\n{exc}")
        _write_log(log_path, "workflow_exit", reason=str(exc))
                
    print()
    if results:
        print("================ Final Agent Responses ================")
        for agent_name, output in results.items():
            print(f"\n[{agent_name}]")
            print(output)
        print("=======================================================")

    for agent_name, output in results.items():
        _write_log(
            log_path,
            "agent_final_output",
            agent_name=agent_name,
            text=output,
        )

    requirements_output = results.get(requirements_agent_name, "")
    if not requirements_output:
        for agent_name, output in results.items():
            normalized_name = agent_name.lower()
            if "requirement" in normalized_name and "extract" in normalized_name:
                requirements_agent_name = agent_name
                requirements_output = output
                break
    if requirements_output:
        requirements_output_path = _save_agent_output_markdown(
            requirements_agent_name or "requirements_extractor",
            requirements_output,
            log_path,
        )
        print(f"Requirements extractor output saved to: {requirements_output_path}")
        await _emit(emit, {"type": "log", "message": f"Requirements extractor output saved to: {requirements_output_path}"})
        _write_log(
            log_path,
            "agent_output_saved",
            agent_name=requirements_agent_name or "requirements_extractor",
            output_path=str(requirements_output_path),
        )

    _write_log(log_path, "workflow_end", agent_count=len(results))
    print(f"Communication log saved to: {log_path}")
    await _emit(emit, {"type": "log", "message": f"Communication log saved to: {log_path}"})
    return results
#"""
