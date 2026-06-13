from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from agent.graph import build_debugger_graph
from agent.state import AgentState

# Initialize the API router
router = APIRouter()

# Compile the LangGraph agent once during startup
debugger_app = build_debugger_graph()

# --- Pydantic Models for Request Validation ---

class DebuggerRequest(BaseModel):
    run_id: str
    epoch: int
    current_metrics: Dict[str, float]
    metric_history: Dict[str, List[float]]

class DebuggerResponse(BaseModel):
    anomalies: List[Dict[str, Any]]
    diagnosis: Optional[str]
    fix_applied: Optional[str]
    verification_status: Optional[str]

# --- Endpoints ---

@router.get("/health", summary="Health Check Endpoint")
def health_check() -> Dict[str, str]:
    """
    Returns the health status of the API.
    """
    return {"status": "healthy", "service": "adaptive-ml-debugger-agent"}

@router.get("/metrics", summary="Prometheus Metrics Endpoint")
def get_metrics() -> Response:
    """
    Exposes real-time training metrics for Prometheus to scrape.
    """
    try:
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate metrics: {str(e)}")

@router.post("/agent/debug", response_model=DebuggerResponse, summary="Trigger Debugger Agent")
def trigger_agent(request: DebuggerRequest) -> Dict[str, Any]:
    """
    Triggers the LangGraph ReAct agent manually.
    Passes the current state (epoch, metrics, history) into the graph to detect anomalies,
    diagnose issues, apply fixes, and verify them.
    """
    try:
        # Construct the initial state for the LangGraph agent
        initial_state: AgentState = {
            "run_id": request.run_id,
            "epoch": request.epoch,
            "current_metrics": request.current_metrics,
            "metric_history": request.metric_history,
            "anomalies": [],
            "diagnosis": None,
            "fix_applied": None,
            "verification_status": None,
            "messages": []
        }

        # Invoke the LangGraph workflow
        result_state = debugger_app.invoke(initial_state)

        # Return the parsed outcome
        return {
            "anomalies": result_state.get("anomalies", []),
            "diagnosis": result_state.get("diagnosis"),
            "fix_applied": result_state.get("fix_applied"),
            "verification_status": result_state.get("verification_status")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")