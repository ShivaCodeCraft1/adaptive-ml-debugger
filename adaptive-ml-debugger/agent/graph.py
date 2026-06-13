from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import monitor_node, detect_node, diagnose_node, fix_node, verify_node

def build_debugger_graph():
    """
    Constructs and compiles the LangGraph StateGraph for the Debugger Agent.
    
    Flow:
    START -> Monitor -> Detect
    Detect -> (If anomalies) -> Diagnose -> Fix -> Verify -> END
    Detect -> (If no anomalies) -> END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("detect", detect_node)
    workflow.add_node("diagnose", diagnose_node)
    workflow.add_node("fix", fix_node)
    workflow.add_node("verify", verify_node)

    # Define edges
    workflow.add_edge(START, "monitor")
    workflow.add_edge("monitor", "detect")

    # Conditional edge routing from Detect
    def route_detect(state: AgentState) -> str:
        anomalies = state.get("anomalies", [])
        if len(anomalies) > 0:
            return "diagnose"
        return END

    workflow.add_conditional_edges(
        "detect",
        route_detect,
        {
            "diagnose": "diagnose",
            END: END
        }
    )

    workflow.add_edge("diagnose", "fix")
    workflow.add_edge("fix", "verify")
    workflow.add_edge("verify", END)

    # Compile the graph
    app = workflow.compile()
    
    return app