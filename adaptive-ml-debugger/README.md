# Adaptive ML Pipeline Debugger Agent

An AI-driven agent designed to monitor Machine Learning training jobs, autonomously detect anomalies, diagnose root causes, and apply remediation steps in real-time. Built with PyTorch, LangGraph, FastAPI, MLflow, and Prometheus.

## Features

1. **Monitor ML Training Jobs**: Tracks training loss, validation loss, accuracy, and gradient norms.
2. **Detect Anomalies**: Identifies exploding/vanishing gradients, train/validation divergence, overfitting, and learning rate instability.
3. **Agentic Workflow**: Uses a LangGraph ReAct agent loop (Monitor → Detect → Diagnose → Fix → Verify → Loop).
4. **Auto-Remediation**: Dynamically reduces learning rate, clips gradients, restores checkpoints, or adjusts model architecture (e.g., adding dropout).
5. **Observability Stack**: Integrated with MLflow for experiment tracking, Prometheus for real-time metrics scraping, and Grafana for dashboards.

## Tech Stack

* **Core**: Python 3.11, PyTorch
* **Agent**: LangGraph, LangChain, OpenAI
* **API**: FastAPI, Uvicorn
* **Observability**: MLflow, Prometheus, Grafana, Weights & Biases
* **Infrastructure**: Docker, Docker Compose, GitHub Actions

## Local Setup

### Prerequisites

* Docker and Docker Compose
* OpenAI API Key (for the LangGraph agent)
* Weights & Biases API Key (optional, for W&B logging)

### Running with Docker Compose

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/adaptive-ml-debugger.git](https://github.com/yourusername/adaptive-ml-debugger.git)
   cd adaptive-ml-debugger