from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    State dictionary for the LangGraph ReAct agent.
    """
    run_id: str
    epoch: int
    current_metrics: Dict[str, float]
    metric_history: Dict[str, List[float]]
    
    anomalies: List[Dict[str, Any]]
    diagnosis: Optional[str]
    fix_applied: Optional[str]
    verification_status: Optional[str]
    
    # LangChain message history, allowing LLMs to converse and retain context
    messages: Annotated[List[BaseMessage], operator.add]