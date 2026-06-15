# 📚 RAG Document Q&A System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

> A production-ready **Retrieval Augmented Generation (RAG)** system that answers questions from your documents using AI.

## ✨ Features

- 📄 **Upload PDF & TXT files** - Create a knowledge base from your documents
- 🔍 **Semantic Search** - Find relevant content using vector embeddings
- 🤖 **AI-Powered Answers** - Get accurate responses from GPT-3.5
- 📖 **Source Attribution** - See exactly which parts of documents were used
- 💬 **Chat Interface** - Natural conversation with your documents
- 🐳 **Docker Support** - Easy deployment anywhere

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-3.5 |
| Vector DB | FAISS |
| Framework | LangChain |
| Embeddings | OpenAI Ada |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RAG-Document-QA-System.git
cd RAG-Document-QA-System

# Run setup script
# Windows:
setup.bat
# Mac/Linux:
bash setup.sh

# Add your OpenAI API key to .env file

# Run the app
streamlit run app.py
```
