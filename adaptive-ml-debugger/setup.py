from setuptools import setup, find_packages

setup(
    name="adaptive-ml-debugger",
    version="0.1.0",
    description="Adaptive ML Pipeline Debugger Agent using LangGraph",
    author="ML Engineering Team",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "langchain>=0.1.0",
        "langgraph>=0.0.24",
        "mlflow>=2.10.0",
        "fastapi>=0.109.0",
        "prometheus-client>=0.19.0",
        "wandb>=0.16.2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.1.0",
            "isort>=5.13.0",
        ]
    },
)