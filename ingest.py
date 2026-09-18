import os
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

DB_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Initial seed URLs for University of Baltistan, Skardu
TARGET_URLS = [
    "https://uobs.edu.pk/",
    "https://uobs.edu.pk/about-us/",
    "https://uobs.edu.pk/academics/",
    "https://uobs.edu.pk/admissions/"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_text(text: str) -> str:
    """Clean and normalize extracted web text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_university_portal() -> list[Document]:
    """Scrape web pages from University of Baltistan portal."""
    documents = []
    print(" starting data collection from UoBS Portal...")
    
    for url in TARGET_URLS:
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Remove non-informative elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                text = soup.get_text(separator=" ")
                cleaned = clean_text(text)
                
                if len(cleaned) > 100:
                    doc = Document(
                        page_content=cleaned,
                        metadata={"source": url, "title": soup.title.string if soup.title else "UoBS Portal"}
                    )
                    documents.append(doc)
                    print(f" Successfully scraped: {url}")
            else:
                print(f" Failed to fetch {url} - Status Code: {response.status_code}")
        except Exception as e:
            print(f" Exception while fetching {url}: {str(e)}")

    # Fallback knowledge base if live site block/timeout occurs during build
    if not documents:
        print(" Using built-in UoBS knowledge fallback dataset...")
        fallback_data = [
            Document(
                page_content="University of Baltistan, Skardu (UoBS) is a higher education institution in Gilgit-Baltistan, Pakistan. It was established in 2017. UoBS offers undergraduate and graduate programs in Computer Science, Business Administration, Environmental Sciences, Biological Sciences, Educational Development, and Humanities.",
                metadata={"source": "uobs_fallback", "title": "UoBS Overview"}
            ),
            Document(
                page_content="UoBS Admissions: Admissions open twice a year for Spring and Fall semesters. Requirements for BS CS include FSC Pre-Engineering or ICS with minimum 50% marks. Scholarships like HEC Need-Based and Ehsaas Undergraduate Scholarships are offered.",
                metadata={"source": "uobs_fallback", "title": "UoBS Admissions"}
            )
        ]
        documents.extend(fallback_data)

    return documents

def build_vector_store():
    """Chunk data, convert to embeddings, and persist in Chroma Vector DB."""
    raw_docs = scrape_university_portal()
    
    # 1. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f" Generated {len(chunks)} text chunks.")

    # 2. Embedding Model setup (Local execution, no API cost)
    print(" Initializing HuggingFace Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 3. Vector Database Indexing
    print(" Saving chunks into Chroma DB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(" Vector Database built successfully!")

if __name__ == "__main__":
    build_vector_store()
