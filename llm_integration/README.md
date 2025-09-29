# Experimental Physics 3 - Interactive Q&A System

An intelligent question-answering system for the Experimental Physics 3 course that uses embeddings and language models to provide instant answers from course materials.

## 🎯 Features

- **Intelligent Q&A**: Ask questions in natural language and get relevant answers from course content
- **Source Attribution**: Every answer includes references to the specific course materials used
- **Confidence Scoring**: See how confident the system is in its answers
- **Web Integration**: Seamlessly integrates with Quarto documents and websites
- **Local Processing**: All data processing happens locally for privacy
- **Multiple Data Sources**: Can index both local Quarto files and published websites

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js and Quarto (for website generation)
- 2-4 GB of free disk space
- 4 GB RAM minimum (8 GB recommended)

## 🚀 Quick Start

### 1. Clone or Download the Repository

```bash
cd "/Users/fci/Library/CloudStorage/Dropbox/work/teaching/EXP3 2025/llm_integration"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install langchain chromadb sentence-transformers fastapi uvicorn beautifulsoup4
```

### 3. Create the Vector Database

```bash
python website_embedder.py
```

You'll be prompted to choose:
- **Option 1**: Scrape the published website (http://fcichos.github.io/Exp3_2024/)
- **Option 2**: Load local Quarto files from the project directory

### 4. Start the Q&A Server

```bash
python qa_server.py
```

The server will start on `http://localhost:8000`

### 5. Test the System

Open your browser and go to: `http://localhost:8000`

You should see a health check response indicating the system is running.

## 📝 Usage

### Adding to Your Quarto Document

1. Add the Q&A widget to any Quarto document:

```markdown
---
title: "My Physics Lecture"
format: html
---

# Content here...

<div class="qa-widget"></div>

<script src="./llm_integration/qa_widget.js"></script>
```

2. Configure the API endpoint (if not using default):

```html
<script>
  window.QA_API_URL = 'http://your-server:8000';
</script>
```

### Using the Pre-built Example

Open `qa_example.qmd` in Quarto and render it:

```bash
quarto render qa_example.qmd
```

Then open the generated HTML file in your browser.

## 🔧 Configuration

### Server Configuration

Edit `qa_server.py` to customize:

```python
# Change the port
run_server(host="0.0.0.0", port=8000)

# Change the embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Change the LLM model
model_name = "google/flan-t5-small"
```

### Widget Configuration

Edit `qa_widget.js` to customize:

```javascript
const CONFIG = {
    apiUrl: 'http://localhost:8000',
    maxRetries: 3,
    defaultContextSize: 5,
    defaultTemperature: 0.7
};
```

### Embedder Configuration

Edit `website_embedder.py` to customize:

```python
embedder = WebsiteEmbedder(
    base_url="http://fcichos.github.io/Exp3_2024/",
    persist_directory="./chroma_db_exp3",
    chunk_size=1000,
    chunk_overlap=200
)
```

## 🏗️ Architecture

```
┌─────────────────────┐
│   Quarto Document   │
│   with Q&A Widget   │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│    FastAPI Server   │
│    (qa_server.py)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Vector Database   │
│     (ChromaDB)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Course Materials   │
│  (.qmd files/web)   │
└─────────────────────┘
```

## 📚 API Documentation

### Health Check
```http
GET /
```

### Ask Question
```http
POST /ask
Content-Type: application/json

{
  "question": "What is wave optics?",
  "context_size": 5,
  "temperature": 0.7,
  "max_tokens": 500
}
```

Response:
```json
{
  "answer": "Wave optics is...",
  "sources": [
    {
      "content": "...",
      "source": "wave-optics/index.qmd",
      "title": "Wave Optics",
      "score": 0.89
    }
  ],
  "timestamp": "2024-01-20T10:30:00",
  "confidence": 0.85
}
```

### List Sources
```http
GET /sources
```

### Submit Feedback
```http
POST /feedback
Content-Type: application/json

{
  "question": "What is wave optics?",
  "answer": "...",
  "helpful": true,
  "comments": "Very clear explanation"
}
```

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError"
```bash
# Install missing module
pip install [module_name]
```

#### "Server not responding"
- Check if the server is running: `python qa_server.py`
- Verify the port is not blocked by firewall
- Check the API URL in your Quarto document

#### "No documents found"
- Run the embedder first: `python website_embedder.py`
- Check that the vector database directory exists
- Verify the source documents are accessible

#### "Low quality answers"
- Increase the context_size parameter
- Rebuild the vector database with smaller chunk_size
- Consider using a more powerful LLM model

### Performance Optimization

For better performance:

1. **Use GPU acceleration** (if available):
```python
model_kwargs={'device': 'cuda'}
```

2. **Use a faster vector database**:
```bash
pip install faiss-gpu  # or faiss-cpu
```

3. **Cache frequent queries**:
```python
from functools import lru_cache
@lru_cache(maxsize=100)
```

## 🔐 Security Considerations

- The server runs locally by default
- No data is sent to external services (unless configured)
- API endpoints should be secured in production:
  - Add authentication
  - Use HTTPS
  - Implement rate limiting

## 🚀 Advanced Usage

### Using OpenAI GPT Models

1. Install OpenAI package:
```bash
pip install openai
```

2. Modify `qa_server.py`:
```python
from langchain.llms import OpenAI
llm = OpenAI(
    temperature=0.7,
    openai_api_key="your-api-key"
)
```

### Using Local Ollama Models

1. Install Ollama: https://ollama.ai

2. Pull a model:
```bash
ollama pull llama2
```

3. Modify `qa_server.py`:
```python
from langchain.llms import Ollama
llm = Ollama(
    model="llama2",
    base_url="http://localhost:11434"
)
```

### Deploying to Production

1. **Use a production ASGI server**:
```bash
gunicorn qa_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Set up reverse proxy** (nginx):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **Use environment variables**:
```bash
export QA_API_KEY="your-secret-key"
export QA_DB_PATH="/path/to/db"
```

## 📊 Monitoring and Analytics

Track usage and performance:

```python
# In qa_server.py
import logging
logging.basicConfig(
    filename='qa_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
```

## 🤝 Contributing

1. Test your changes thoroughly
2. Update documentation
3. Follow Python PEP 8 style guidelines
4. Add unit tests for new features

## 📄 License

This project is part of the Experimental Physics 3 course materials.

## 👤 Contact

For questions or issues related to this Q&A system:
- Course Instructor: Prof. Dr. Frank Cichos
- Technical Support: [Create an issue in the repository]

## 🔄 Updates and Maintenance

### Updating Course Content

1. Add or modify `.qmd` files
2. Re-run the embedder:
```bash
python website_embedder.py
```
3. Restart the server

### Updating Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Backing Up the Database

```bash
cp -r chroma_db_exp3 chroma_db_exp3_backup
```

## 📈 Performance Metrics

Typical performance on standard hardware:
- Embedding creation: ~100 documents/minute
- Query response time: 1-3 seconds
- Memory usage: 500MB - 2GB
- Disk space: 200MB - 1GB (depending on content size)

## 🎓 Educational Resources

Learn more about the technologies used:
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Guide](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)
- [Quarto Documentation](https://quarto.org/)

---

Made with ❤️ for the Experimental Physics 3 course at Leipzig University