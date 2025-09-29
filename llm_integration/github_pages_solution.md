# GitHub Pages Compatible Q&A Solutions

Since GitHub Pages only serves static content, here are several approaches to add Q&A functionality to your course website hosted on GitHub Pages.

## Solution 1: Client-Side Search with Pre-computed Embeddings

This approach pre-computes all embeddings and serves them as static JSON files.

### Step 1: Generate Static Embeddings Database

Create `generate_static_embeddings.py`:

```python
"""
Generate static embeddings for GitHub Pages
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers", "scikit-learn"])
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

class StaticEmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []

    def load_quarto_files(self, directory: str):
        """Load all .qmd files from directory"""
        qmd_files = Path(directory).rglob("*.qmd")

        for qmd_file in qmd_files:
            try:
                with open(qmd_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Remove YAML frontmatter
                    if content.startswith('---'):
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            content = parts[2]

                    # Split into chunks
                    chunks = self._chunk_text(content, chunk_size=500)

                    for i, chunk in enumerate(chunks):
                        self.documents.append({
                            'id': f"{qmd_file.stem}_{i}",
                            'content': chunk,
                            'source': str(qmd_file.name),
                            'title': qmd_file.stem.replace('-', ' ').title()
                        })

            except Exception as e:
                print(f"Error loading {qmd_file}: {e}")

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) > 50:  # Minimum chunk size
                chunks.append(chunk)

        return chunks

    def generate_embeddings(self):
        """Generate embeddings for all documents"""
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

    def save_to_json(self, output_dir: str = "_site/embeddings"):
        """Save embeddings and documents as JSON files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Convert embeddings to list for JSON serialization
        embeddings_list = self.embeddings.tolist()

        # Split into smaller files for better performance
        chunk_size = 100
        for i in range(0, len(self.documents), chunk_size):
            chunk_docs = self.documents[i:i + chunk_size]
            chunk_embeddings = embeddings_list[i:i + chunk_size]

            chunk_data = {
                'documents': chunk_docs,
                'embeddings': chunk_embeddings
            }

            output_file = Path(output_dir) / f"embeddings_{i // chunk_size}.json"
            with open(output_file, 'w') as f:
                json.dump(chunk_data, f)

        # Save metadata
        metadata = {
            'total_documents': len(self.documents),
            'chunks': (len(self.documents) + chunk_size - 1) // chunk_size,
            'model': "sentence-transformers/all-MiniLM-L6-v2",
            'embedding_dim': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
        }

        with open(Path(output_dir) / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(self.documents)} documents to {output_dir}")

# Usage
if __name__ == "__main__":
    generator = StaticEmbeddingGenerator()
    generator.load_quarto_files(".")  # Current directory
    generator.generate_embeddings()
    generator.save_to_json("_site/embeddings")
```

### Step 2: Client-Side Search JavaScript

Create `static_qa_widget.js`:

```javascript
/**
 * Static Q&A Widget for GitHub Pages
 * Uses pre-computed embeddings for client-side similarity search
 */

class StaticQAWidget {
    constructor(containerId, embeddingsPath = './embeddings') {
        this.container = document.getElementById(containerId);
        this.embeddingsPath = embeddingsPath;
        this.documents = [];
        this.embeddings = [];
        this.metadata = null;
        this.model = null;

        this.initialize();
    }

    async initialize() {
        // Load metadata
        const metadataResponse = await fetch(`${this.embeddingsPath}/metadata.json`);
        this.metadata = await metadataResponse.json();

        // Load all embedding chunks
        for (let i = 0; i < this.metadata.chunks; i++) {
            const response = await fetch(`${this.embeddingsPath}/embeddings_${i}.json`);
            const data = await response.json();
            this.documents.push(...data.documents);
            this.embeddings.push(...data.embeddings);
        }

        // Load the sentence transformer model for query encoding
        await this.loadModel();

        // Render the widget
        this.render();
    }

    async loadModel() {
        // Use TensorFlow.js with Universal Sentence Encoder
        if (typeof tf === 'undefined') {
            // Load TensorFlow.js
            await this.loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
            await this.loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/universal-sentence-encoder');
        }

        // Load the model
        this.model = await use.load();
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async search(query, topK = 5) {
        if (!this.model) {
            throw new Error('Model not loaded yet');
        }

        // Encode the query
        const queryEmbedding = await this.model.embed([query]);
        const queryArray = await queryEmbedding.array();

        // Calculate cosine similarity with all documents
        const similarities = this.embeddings.map(docEmbedding =>
            this.cosineSimilarity(queryArray[0], docEmbedding)
        );

        // Get top K results
        const results = similarities
            .map((score, index) => ({
                score,
                document: this.documents[index]
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);

        return results;
    }

    cosineSimilarity(a, b) {
        const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
        const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
        const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
        return dotProduct / (normA * normB);
    }

    render() {
        this.container.innerHTML = `
            <div class="static-qa-widget">
                <h3>📚 Ask a Question</h3>
                <div class="qa-input-group">
                    <input type="text" id="qa-input" placeholder="Enter your question...">
                    <button id="qa-submit">Search</button>
                </div>
                <div id="qa-loading" style="display:none;">Searching...</div>
                <div id="qa-results"></div>
            </div>
        `;

        // Add styles
        this.addStyles();

        // Add event listeners
        document.getElementById('qa-submit').addEventListener('click', () => this.handleSearch());
        document.getElementById('qa-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleSearch();
        });
    }

    async handleSearch() {
        const query = document.getElementById('qa-input').value.trim();
        if (!query) return;

        const loading = document.getElementById('qa-loading');
        const resultsDiv = document.getElementById('qa-results');

        loading.style.display = 'block';
        resultsDiv.innerHTML = '';

        try {
            const results = await this.search(query, 5);

            resultsDiv.innerHTML = `
                <h4>Top Results:</h4>
                ${results.map(r => `
                    <div class="qa-result">
                        <div class="qa-result-title">${r.document.title}</div>
                        <div class="qa-result-content">${r.document.content.substring(0, 200)}...</div>
                        <div class="qa-result-score">Relevance: ${(r.score * 100).toFixed(1)}%</div>
                    </div>
                `).join('')}
            `;
        } catch (error) {
            resultsDiv.innerHTML = `<div class="qa-error">Error: ${error.message}</div>`;
        } finally {
            loading.style.display = 'none';
        }
    }

    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .static-qa-widget {
                max-width: 800px;
                margin: 2rem auto;
                padding: 1.5rem;
                border: 1px solid #ddd;
                border-radius: 8px;
            }

            .qa-input-group {
                display: flex;
                gap: 1rem;
                margin: 1rem 0;
            }

            #qa-input {
                flex: 1;
                padding: 0.5rem;
                border: 1px solid #ccc;
                border-radius: 4px;
            }

            #qa-submit {
                padding: 0.5rem 1rem;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }

            #qa-submit:hover {
                background: #0056b3;
            }

            .qa-result {
                padding: 1rem;
                margin: 0.5rem 0;
                background: #f8f9fa;
                border-left: 3px solid #007bff;
                border-radius: 4px;
            }

            .qa-result-title {
                font-weight: bold;
                color: #333;
            }

            .qa-result-content {
                margin: 0.5rem 0;
                color: #666;
            }

            .qa-result-score {
                font-size: 0.9em;
                color: #007bff;
            }
        `;
        document.head.appendChild(style);
    }
}
```

## Solution 2: Use Free API Services

### Option A: Hugging Face Inference API (Free Tier)

Create `hf_qa_widget.js`:

```javascript
/**
 * Q&A Widget using Hugging Face Inference API
 */

class HuggingFaceQAWidget {
    constructor(containerId, apiKey) {
        this.container = document.getElementById(containerId);
        this.apiKey = apiKey;
        this.apiUrl = 'https://api-inference.huggingface.co/models/';
        this.embeddingModel = 'sentence-transformers/all-MiniLM-L6-v2';
        this.qaModel = 'deepset/roberta-base-squad2';

        this.render();
    }

    async embed(texts) {
        const response = await fetch(this.apiUrl + this.embeddingModel, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                inputs: texts
            })
        });

        return await response.json();
    }

    async answerQuestion(question, context) {
        const response = await fetch(this.apiUrl + this.qaModel, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                inputs: {
                    question: question,
                    context: context
                }
            })
        });

        return await response.json();
    }

    render() {
        // Similar to static widget but using HF API
    }
}
```

### Option B: Cloudflare Workers (Free Tier)

Create `cloudflare-worker.js`:

```javascript
/**
 * Deploy this as a Cloudflare Worker for the API backend
 */

addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
    // Enable CORS
    const headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }

    if (request.method === 'OPTIONS') {
        return new Response(null, { headers })
    }

    if (request.method === 'POST' && request.url.includes('/search')) {
        const { question } = await request.json()

        // Use Cloudflare AI (available in Workers)
        const embeddings = await env.AI.run(
            '@cf/baai/bge-base-en-v1.5',
            { text: question }
        )

        // Search your pre-stored embeddings in KV store
        const results = await searchEmbeddings(embeddings)

        return new Response(JSON.stringify(results), { headers })
    }

    return new Response('Not found', { status: 404 })
}
```

## Solution 3: Hybrid Approach - GitHub Actions + GitHub Pages

Use GitHub Actions to periodically update embeddings:

### `.github/workflows/update-embeddings.yml`:

```yaml
name: Update Embeddings

on:
  push:
    paths:
      - '**.qmd'
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install sentence-transformers scikit-learn

    - name: Generate embeddings
      run: |
        python generate_static_embeddings.py

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./_site
```

## Solution 4: Use Vercel/Netlify Functions (Free Tier)

### Vercel Function (`api/search.js`):

```javascript
import { Configuration, OpenAIApi } from 'openai';

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { question } = req.body;

    // Use OpenAI API or any other service
    const configuration = new Configuration({
        apiKey: process.env.OPENAI_API_KEY,
    });

    const openai = new OpenAIApi(configuration);

    try {
        const response = await openai.createEmbedding({
            model: "text-embedding-ada-002",
            input: question,
        });

        // Search logic here

        res.status(200).json({ answer: "..." });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
}
```

## Implementation Guide

### For Immediate GitHub Pages Deployment:

1. **Generate static embeddings**:
```bash
python generate_static_embeddings.py
```

2. **Add to your Quarto document**:
```html
<div id="qa-widget"></div>

<script>
    // Initialize the static Q&A widget
    document.addEventListener('DOMContentLoaded', function() {
        new StaticQAWidget('qa-widget', './embeddings');
    });
</script>
<script src="./static_qa_widget.js"></script>
```

3. **Deploy to GitHub Pages**:
```bash
git add .
git commit -m "Add Q&A widget"
git push
```

### Pros and Cons of Each Solution:

| Solution | Pros | Cons |
|----------|------|------|
| **Static Embeddings** | No API needed, Fast, Free | Limited to similarity search |
| **Hugging Face API** | Advanced models, Free tier | Rate limits, Requires API key |
| **Cloudflare Workers** | Fast edge compute, Generous free tier | Some setup complexity |
| **GitHub Actions** | Automated updates, Free | Build time limits |
| **Vercel/Netlify** | Serverless functions, Easy deploy | Cold starts, Limits on free tier |

## Recommended Approach

For your use case, I recommend:

1. **Start with Static Embeddings** (Solution 1) for immediate deployment
2. **Add Hugging Face API** (Solution 2A) for better answers
3. **Use GitHub Actions** (Solution 3) to keep embeddings updated

This gives you a working solution immediately while allowing for improvements over time.

## Security Notes

When using API keys in client-side code:
- Use environment variables in build process
- Consider using a proxy service
- Implement rate limiting
- Monitor usage

## Testing Locally

```bash
# Serve your site locally
python -m http.server 8000

# Or use any static server
npx serve .
```

Then open http://localhost:8000 in your browser.
