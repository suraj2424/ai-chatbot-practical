# Project Description

This project implements an agent-based system powered by **LangGraph** designed to facilitate complex, multi-step tasks involving file system interaction and code execution.

## Key Features

*   **Agent Workflow (LangGraph):** Defines a stateful graph for orchestrating agent behavior, allowing for conditional routing based on tool outputs.
*   **File System Tools:** Implements custom tools for secure interaction with the local workspace, including:
    *   `read_file`: Reading content from specified files.
    *   `write_file`: Writing content to files.
    *   `list_files_in_workspace`: Listing all files within the working directory.
    *   `lint_code`: Checking Python syntax errors.
    *   `execute_code`: Running arbitrary Python code within the workspace.
*   **LLM Integration:** Integrates a Large Language Model (via Ollama) to reason about user requests and decide which tools to use.
*   **Persistence:** Uses `MemorySaver` for state management and checkpointing of the graph execution.

## Technology Stack

*   Python
*   LangChain / LangGraph
*   Ollama (for LLM)
*   `dotenv`
*   `pathlib` and `subprocess` for system interaction.

## How it Works

The system defines a state and a graph structure where an LLM decides the next step. If the LLM requests an action that requires file access or code execution, the workflow routes to the appropriate tool execution node.

## Setup

1.  **Dependencies:** Install required Python packages (e.g., `langchain`, `langgraph`, etc.).
2.  **Environment:** Ensure you have an Ollama instance running with a suitable model (e.g., `gemma4:e2b`).
3.  **Environment Variables:** Set up necessary environment variables (e.g., for API keys, if applicable).

## Usage

This project serves as a framework for building sophisticated agents that can perform tasks requiring both natural language understanding and concrete file/code manipulation.