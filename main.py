"""
AI-Driven RFP Generator - FastAPI Backend
US-01.1  Create RFP Workspace / Case ID
US-01.2  Upload Multiple RFP Documents
US-01.3  Validate Uploaded File Type and Integrity
US-01.4  Detect and Manage Duplicate Documents
US-01.5  Show Complete Ingestion Status and Failure Summary
"""

import asyncio, json, re, uuid, webbrowser, threading
from pathlib import Path
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.chatbot import answer_query_from_vector_db
from app.create_vectordb import create_vector_db
from app.run_workflow import run_analysis_workflow
from models import ChatbotRequest, ResolveRequest, WorkflowHumanResponseRequest, WorkflowRunRequest, WorkspaceRequest
from utils.file_utils import byte_hash, content_hash, fmt_size

# Constants
ALLOWED = {".pdf",".docx",".md",".xlsx",".xls",".csv",
           ".png",".jpg",".jpeg",".gif",".bmp",".tiff",".webp"}
BASE = Path.home() / "RFP_Workspaces"
BASE.mkdir(parents=True, exist_ok=True)

# In-memory session state
session: dict = {"workspace": None, "files": {}}   # files: id -> FileInfo dict
workflow_response_queues: dict[str, asyncio.Queue[dict]] = {}

app = FastAPI(title="AI-Driven RFP Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/workspace")
def create_workspace(req: WorkspaceRequest):
    safe = re.sub(r'[<>:"/\\|?*]',"_", req.name).strip()
    if not safe:
        raise HTTPException(400, "Invalid workspace name")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BASE / f"{safe}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    session["workspace"] = str(path)
    session["files"] = {}
    return {"path": str(path), "name": safe}

@app.get("/api/workspace")
def get_workspace():
    return {"workspace": session.get("workspace"), "files": list(session["files"].values())}

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    ws = session.get("workspace")
    if not ws:
        raise HTTPException(400, "No workspace. Create one first.")
    ws_path = Path(ws)
    results = []
    for uf in files:
        ext = Path(uf.filename).suffix.lower()
        if ext not in ALLOWED:
            results.append({"name": uf.filename, "status": "rejected",
                            "reason": f"Format '{ext}' not supported"})
            continue
        data = await uf.read()
        dest = ws_path / uf.filename
        # handle name collisions
        if dest.exists():
            stem = dest.stem
            dest = ws_path / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
        dest.write_bytes(data)
        fid = str(uuid.uuid4())
        bh = byte_hash(dest)
        ch = content_hash(dest)
        info = {
            "id": fid, "name": uf.filename, "saved_name": dest.name,
            "ext": ext[1:].upper(), "size": fmt_size(len(data)),
            "byte_hash": bh, "content_hash": ch,
            "status": "staged", "path": str(dest)
        }
        session["files"][fid] = info
        results.append({"name": uf.filename, "status": "staged", "id": fid})
    return {"results": results}

@app.post("/api/ingest")
def run_ingestion():
    files = list(session["files"].values())
    staged = [f for f in files if f["status"] == "staged"]
    if not staged:
        raise HTTPException(400, "No staged files to ingest.")

    # Duplicate detection across ALL files (ingested + staged)
    seen: dict[str, dict] = {}
    duplicates = []
    for f in files:
        key = f["content_hash"]
        if key in seen:
            duplicates.append({"original": seen[key]["id"], "duplicate": f["id"],
                                "original_name": seen[key]["name"], "duplicate_name": f["name"]})
        else:
            seen[key] = f

    ws = session.get("workspace")
    vector_db = str(Path(ws) / "vectorstore") if ws else None

    return {"duplicates": duplicates,
            "staged_count": len(staged),
            "total_files": len(files),
            "vector_db": vector_db}

@app.post("/api/resolve")
def resolve_duplicate(req: ResolveRequest):
    f = session["files"].get(req.file_id)
    if not f:
        raise HTTPException(404, "File not found")
    if req.action == "delete":
        try: Path(f["path"]).unlink(missing_ok=True)
        except Exception: pass
        f["status"] = "deleted"
    else:
        f["status"] = "kept_duplicate"
    return {"id": req.file_id, "status": f["status"]}

@app.post("/api/finalize")
def finalize_ingestion():
    ws = session.get("workspace")
    if not ws:
        raise HTTPException(400, "No workspace. Create one first.")

    ingested = 0
    for f in session["files"].values():
        if f["status"] == "staged":
            f["status"] = "ingested"
            ingested += 1

    total = sum(1 for f in session["files"].values() if f["status"] == "ingested")
    if total == 0:
        raise HTTPException(400, "No ingested files available to index.")

    ws_path = Path(ws)
    vector_db_path = ws_path / "vectorstore"
    try:
        collection, usage_report_path = create_vector_db(
            upload_dir=ws_path,
            index_path=vector_db_path,
        )
    except Exception as ex:
        raise HTTPException(500, f"Failed creating vector database: {ex}") from ex

    return {"ingested": ingested, "total": total,
            "workspace": ws,
            "vector_db": str(vector_db_path),
            "rag_created": collection is not None,
            "usage_report": str(usage_report_path)}

@app.post("/api/workflow/run")
async def run_workflow(req: WorkflowRunRequest):
    vector_db_path = Path(req.index_path)/"vectorstore"
    if not vector_db_path.exists():
        raise HTTPException(400, f"No vector database found at '{vector_db_path}'.")
    markdown_data_path = Path(req.index_path)/"markdown"
    if not markdown_data_path.exists():
        raise HTTPException(400, f"No markdown data found at '{markdown_data_path}'.")

    try:
        workflow_output = await run_analysis_workflow(
            index_path=vector_db_path,
            markdown_data_path = markdown_data_path
        )
    except Exception as ex:
        raise HTTPException(500, f"Failed running workflow: {ex}") from ex

    if workflow_output is None:
        raise HTTPException(500, "Workflow did not return results.")

    return {
        "workspace": str(vector_db_path.parent),
        "vector_db": str(vector_db_path),
        "results": workflow_output["results"],
        "usage_report": workflow_output["usage_report"],
    }

def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True, default=str)}\n\n"

@app.post("/api/workflow/run/stream")
async def run_workflow_stream(req: WorkflowRunRequest):
    run_id = uuid.uuid4().hex
    human_response_queue: asyncio.Queue[dict] = asyncio.Queue()
    workflow_response_queues[run_id] = human_response_queue

    vector_db_path = Path(req.index_path) / "vectorstore"
    if not vector_db_path.exists():
        workflow_response_queues.pop(run_id, None)
        raise HTTPException(400, f"No vector database found at '{vector_db_path}'.")

    markdown_data_path = Path(req.index_path) / "markdown"
    if not markdown_data_path.exists():
        workflow_response_queues.pop(run_id, None)
        raise HTTPException(400, f"No markdown data found at '{markdown_data_path}'.")

    async def event_generator():
        ui_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def emit(event: dict) -> None:
            event.setdefault("run_id", run_id)
            await ui_queue.put(event)

        async def run_background() -> None:
            try:
                await emit({"type": "start", "message": "Workflow stream started."})
                workflow_output = await run_analysis_workflow(
                    index_path=vector_db_path,
                    markdown_data_path=markdown_data_path,
                    emit=emit,
                    human_response_queue=human_response_queue,
                )
                await emit({
                    "type": "final_results",
                    "workspace": str(vector_db_path.parent),
                    "vector_db": str(vector_db_path),
                    "results": workflow_output["results"],
                    "usage_report": workflow_output["usage_report"],
                })
                await emit({"type": "done"})
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                await emit({"type": "error", "message": str(ex)})
            finally:
                workflow_response_queues.pop(run_id, None)

        task = asyncio.create_task(run_background())
        try:
            while True:
                event = await ui_queue.get()
                yield _sse_event(event)
                if event.get("type") in {"done", "error"}:
                    break
        finally:
            task.cancel()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/workflow/respond")
async def workflow_respond(req: WorkflowHumanResponseRequest):
    queue = workflow_response_queues.get(req.run_id)
    if queue is None:
        raise HTTPException(404, "Workflow run not found or already finished.")

    await queue.put({
        "request_id": req.request_id,
        "action": req.action,
        "text": req.text,
    })
    return {"ok": True}

@app.post("/api/chatbot")
async def chatbot(req: ChatbotRequest):
    ws = session.get("workspace")
    if not ws:
        raise HTTPException(400, "No workspace. Create one first.")

    vector_db_path = Path(ws) / "vectorstore"
    if not vector_db_path.exists():
        raise HTTPException(400, f"No vector database found at '{vector_db_path}'. Run finalize first.")

    try:
        print(f"Answering question: '{req.question}' using vector DB at '{vector_db_path}'")
        answer = await answer_query_from_vector_db(
            query=req.question,
            index_path=vector_db_path,
            top_k=3,
        )
    except Exception as ex:
        raise HTTPException(500, f"Failed answering chatbot query: {ex}") from ex

    return {
        "question": req.question,
        "answer": answer,
        "workspace": ws,
        "vector_db": str(vector_db_path),
    }

@app.get("/api/files")
def list_files():
    return {"files": list(session["files"].values())}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
