# AI Researcher Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent agent framework designed specifically for academic research assistance. It can search for papers on arXiv, read and parse PDFs, maintain a vector-based knowledge base (RAG), and assist with academic writing and polishing.

## ✨ Features

- **Modular Architecture**: Clean separation of core, LLM adapters, memory, skills, and orchestration.
- **Multi-LLM Support**: OpenAI, Anthropic Claude, and local models via Ollama.
- **Academic Skills**: Built-in skills for arXiv fetching, PDF reading, and paper writing/polishing.
- **RAG Memory**: Store papers in a vector database and retrieve them semantically.
- **Flexible Orchestration**: Choose from ReAct, Plan-and-Execute, or LLMCompiler loops. Graph-based workflows also supported.
- **Easy Extension**: Add custom skills by creating a class or a Markdown file.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-researcher-assistant.git
cd ai-researcher-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
