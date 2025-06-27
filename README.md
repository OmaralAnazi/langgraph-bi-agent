# LangGraph BI Agent

A simple local agent that can access a database and generate reports. This project is for learning and practicing scalable agent design with [LangGraph](https://github.com/langchain-ai/langgraph).

## Features
- Local LLM via [Ollama](https://ollama.com/)
- Database access
- Report generation

## Setup
1. **Install prerequisites:**
   - Python 3.8+
   - [Ollama](https://ollama.com/download)
   - [uv](https://github.com/astral-sh/uv)

2. **Start Ollama and pull a model:**
   ```sh
   ollama pull qwen2:7b
   ollama run qwen2:7b
   ```

3. **Install dependencies:**
   ```sh
   uv pip install .
   ```

## Run
```sh
python main.py
```

---

This project is for practice and learning. Tweak, break, and scale as you wish!
