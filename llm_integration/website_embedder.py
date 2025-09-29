"""
Website Content Embedder for Experimental Physics 3 Course
This script embeds website content into a vector database for LLM-based Q&A
"""

import os
import json
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# Vector database and embedding libraries
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.schema import Document
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "langchain", "chromadb", "sentence-transformers", "beautifulsoup4"])
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.schema import Document


class WebsiteEmbedder:
    """
    A class to scrape website content and create embeddings for Q&A
    """

    def __init__(self,
                 base_url: str = "http://fcichos.github.io/Exp3_2024/",
                 persist_directory: str = "./chroma_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the WebsiteEmbedder

        Args:
            base_url: The base URL of the website to scrape
            persist_directory: Directory to persist the vector database
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.base_url = base_url
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.visited_urls = set()
        self.documents = []

        # Initialize the embedding model (using a lightweight model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

    def scrape_website(self, max_pages: int = 100) -> List[Document]:
        """
        Recursively scrape the website starting from base_url

        Args:
            max_pages: Maximum number of pages to scrape

        Returns:
            List of Document objects containing scraped content
        """
        print(f"Starting to scrape {self.base_url}")
        self._scrape_page(self.base_url, max_pages)
        print(f"Scraped {len(self.documents)} pages")
        return self.documents

    def _scrape_page(self, url: str, max_pages: int):
        """
        Recursively scrape a single page and follow links
        """
        if len(self.visited_urls) >= max_pages:
            return

        if url in self.visited_urls:
            return

        # Mark as visited
        self.visited_urls.add(url)

        try:
            print(f"Scraping: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract main content (adjust selectors based on your site structure)
            # Try multiple possible content containers
            content_selectors = [
                'main',
                'article',
                '.content',
                '#content',
                '.main-content',
                'div.quarto-content',
                'section.level1',
                'body'
            ]

            content = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem
                    break

            if not content:
                content = soup.body

            # Extract text content
            text = content.get_text(separator='\n', strip=True)

            # Extract title
            title = soup.find('title')
            title_text = title.string if title else urlparse(url).path

            # Create document
            if text and len(text) > 100:  # Only add if there's substantial content
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': url,
                        'title': title_text,
                        'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                )
                self.documents.append(doc)

            # Find and follow internal links
            base_domain = urlparse(self.base_url).netloc
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                parsed_url = urlparse(absolute_url)

                # Only follow internal links and avoid fragments/anchors
                if (parsed_url.netloc == base_domain and
                    not parsed_url.fragment and
                    absolute_url not in self.visited_urls and
                    not absolute_url.endswith(('.pdf', '.zip', '.png', '.jpg', '.jpeg', '.gif'))):

                    # Add small delay to be respectful
                    time.sleep(0.5)
                    self._scrape_page(absolute_url, max_pages)

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")

    def load_local_content(self, directory_path: str = None) -> List[Document]:
        """
        Load content from local Quarto files instead of scraping

        Args:
            directory_path: Path to the directory containing .qmd files

        Returns:
            List of Document objects
        """
        if directory_path is None:
            directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        documents = []

        # Find all .qmd files
        qmd_files = Path(directory_path).rglob("*.qmd")

        for qmd_file in qmd_files:
            try:
                with open(qmd_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Remove YAML frontmatter if present
                    if content.startswith('---'):
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            content = parts[2]

                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': str(qmd_file),
                            'title': qmd_file.stem.replace('-', ' ').title(),
                            'file_type': 'qmd'
                        }
                    )
                    documents.append(doc)
                    print(f"Loaded: {qmd_file.name}")

            except Exception as e:
                print(f"Error loading {qmd_file}: {str(e)}")

        self.documents.extend(documents)
        return documents

    def create_embeddings(self, documents: List[Document] = None) -> Chroma:
        """
        Create embeddings from documents and store in vector database

        Args:
            documents: List of documents to embed (uses self.documents if None)

        Returns:
            Chroma vector store instance
        """
        if documents is None:
            documents = self.documents

        if not documents:
            raise ValueError("No documents to embed. Run scrape_website() or load_local_content() first.")

        print(f"Creating embeddings for {len(documents)} documents...")

        # Split documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

        print(f"Split into {len(all_chunks)} chunks")

        # Create or load vector store
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )

        # Persist the database
        vectorstore.persist()
        print(f"Vector database persisted to {self.persist_directory}")

        return vectorstore

    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk

        Returns:
            Chroma vector store instance
        """
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def query(self, question: str, k: int = 5) -> List[Dict]:
        """
        Query the vector database

        Args:
            question: The question to search for
            k: Number of relevant documents to return

        Returns:
            List of relevant documents with scores
        """
        vectorstore = self.load_vectorstore()

        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(question, k=k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content[:500],  # Truncate for display
                'source': doc.metadata.get('source', 'Unknown'),
                'title': doc.metadata.get('title', 'Unknown'),
                'score': float(score)
            })

        return formatted_results


def main():
    """
    Main function to run the embedding pipeline
    """
    # Initialize embedder
    embedder = WebsiteEmbedder(
        base_url="http://fcichos.github.io/Exp3_2024/",
        persist_directory="./chroma_db_exp3"
    )

    # Check if vector database already exists
    db_exists = os.path.exists(embedder.persist_directory)

    if not db_exists or input("Rebuild vector database? (y/n): ").lower() == 'y':
        # Option 1: Scrape website
        print("\nOption 1: Scrape website")
        print("Option 2: Load local Quarto files")
        choice = input("Choose option (1/2): ")

        if choice == "1":
            # Scrape the website
            embedder.scrape_website(max_pages=50)
        else:
            # Load local content
            embedder.load_local_content()

        # Create embeddings
        if embedder.documents:
            vectorstore = embedder.create_embeddings()
            print(f"\nSuccessfully created embeddings for {len(embedder.documents)} documents")

    # Test querying
    print("\n" + "="*50)
    print("Testing Q&A System")
    print("="*50)

    test_questions = [
        "What is geometrical optics?",
        "Explain wave optics",
        "What are electromagnetic waves?",
        "Tell me about quantum mechanics"
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        results = embedder.query(question, k=3)
        print(f"Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. Source: {result['title']}")
            print(f"     Score: {result['score']:.4f}")
            print(f"     Preview: {result['content'][:150]}...")


if __name__ == "__main__":
    main()
