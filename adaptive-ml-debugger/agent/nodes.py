import os
import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent.state import AgentState
from agent.prompts import DIAGNOSE_SYSTEM_PROMPT, FIX_SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT

from anomaly_detection.exploding_gradients import detect_exploding_gradients
from anomaly_detection.vanishing_gradients import detect_vanishing_gradients
from anomaly_detection.divergence import detect_divergence
from anomaly_detection.overfitting import detect_overfitting

from tools.modify_lr import modify_learning_rate
from tools.rollback_checkpoint import rollback_checkpoint
from tools.architecture_fix import suggest_architecture_fix

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-init")
)

# Bind tools to the LLM for the Fix node
tools = [modify_learning_rate, rollback_checkpoint, suggest_architecture_fix]
llm_with_tools = llm.bind_tools(tools)

def monitor_node(state: AgentState) -> AgentState:
    """
    Monitor node: Gathers the latest metrics and history.
    In a real runtime, this might poll MLflow or Prometheus. 
    Here, it ensures the state is ready for detection.
    """
    # Simply pass through the state as data is injected by the training loop
    return state

def detect_node(state: AgentState) -> AgentState:
    """
    Detect node: Runs deterministic heuristics against metric history 
    to identify vanishing/exploding gradients, divergence, or overfitting.
    """
    history = state.get("metric_history", {})
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    grad_norms = history.get("avg_grad_norm", [])

    anomalies = []

    # Run detectors
    exp_grad = detect_exploding_gradients(grad_norms)
    if exp_grad["detected"]:
        anomalies.append({"type": "Exploding Gradients", "message": exp_grad["message"]})

    van_grad = detect_vanishing_gradients(grad_norms)
    if van_grad["detected"]:
        anomalies.append({"type": "Vanishing Gradients", "message": van_grad["message"]})

    div = detect_divergence(train_losses, val_losses)
    if div["detected"]:
        anomalies.append({"type": "Divergence", "message": div["message"]})

    overfit = detect_overfitting(train_losses, val_losses)
    if overfit["detected"]:
        anomalies.append({"type": "Overfitting", "message": overfit["message"]})

    return {"anomalies": anomalies}

def diagnose_node(state: AgentState) -> AgentState:
    """
    Diagnose node: Uses an LLM to analyze the detected anomalies and current metrics.
    """
    prompt = DIAGNOSE_SYSTEM_PROMPT.format(
        epoch=state.get("epoch"),
        current_metrics=state.get("current_metrics"),
        anomalies=state.get("anomalies")
    )
    
    response = llm.invoke([SystemMessage(content=prompt)])
    diagnosis = str(response.content)
    
    return {
        "diagnosis": diagnosis,
        "messages": [AIMessage(content=f"Diagnosis: {diagnosis}")]
    }

def fix_node(state: AgentState) -> AgentState:
    """
    Fix node: LLM selects and executes a remediation tool based on the diagnosis.
    """
    prompt = FIX_SYSTEM_PROMPT.format(
        diagnosis=state.get("diagnosis", "No diagnosis available.")
    )
    
    # Ask LLM to choose a tool
    response = llm_with_tools.invoke([SystemMessage(content=prompt)])
    
    fix_applied = "No tool called. LLM suggested: " + str(response.content)
    
    # Manually execute the tool call if the LLM made one
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Execute corresponding tool
        if tool_name == "modify_learning_rate":
            tool_res = modify_learning_rate.invoke(tool_args)
        elif tool_name == "rollback_checkpoint":
            tool_res = rollback_checkpoint.invoke(tool_args)
        elif tool_name == "suggest_architecture_fix":
            tool_res = suggest_architecture_fix.invoke(tool_args)
        else:
            tool_res = f"Unknown tool: {tool_name}"
            
        fix_applied = f"Executed {tool_name} with args {tool_args}. Result: {tool_res}"

    return {
        "fix_applied": fix_applied,
        "messages": [AIMessage(content=f"Fix Applied: {fix_applied}")]
    }

def verify_node(state: AgentState) -> AgentState:
    """
    Verify node: LLM checks if the applied fix stabilized the system.
    """
    prompt = VERIFY_SYSTEM_PROMPT.format(
        fix_applied=state.get("fix_applied", "None"),
        current_metrics=state.get("current_metrics"),
        anomalies=state.get("anomalies")
    )
    
    response = llm.invoke([SystemMessage(content=prompt)])
    verification = str(response.content)
    
    return {
        "verification_status": verification,
        "messages": [AIMessage(content=f"Verification: {verification}")]
    }