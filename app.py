import os
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Page Configuration
st.set_page_config(
    page_title="UoBS AI Assistant",
    page_icon="🎓",
    layout="wide"
)

DB_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "llama-3.3-70b-versatile"  # Text Generation model on Groq

# Initialize Groq Client safely from Streamlit secrets or system env
api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

@st.cache_resource
def load_vector_db():
    """Load cached ChromaDB vector store."""
    if not os.path.exists(DB_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def retrieve_context(query: str, db: Chroma, top_k: int = 3) -> str:
    """Search Vector Database for relevant chunks."""
    if db is None:
        return ""
    docs = db.similarity_search(query, k=top_k)
    context = "\n\n".join([f"[Source: {doc.metadata.get('source', 'UoBS Data')}]\n{doc.page_content}" for doc in docs])
    return context

# Streamlit App UI
st.title("🎓 University of Baltistan, Skardu")
st.subheader("AI Campus Knowledge Assistant")

if not api_key:
    st.error("🔑 Groq API Key is missing! Add `GROQ_API_KEY` to your Streamlit secrets or environment variables.")
    st.stop()

# Initialize Groq Client
client = Groq(api_key=api_key)

# Load Chroma DB
db = load_vector_db()

if db is None:
    st.warning("⚠️ Vector DB not found. Please run `python ingest.py` locally before deploying to populate knowledge base.")

# Initialize Chat Session History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the UoBS AI Assistant. How can I help you regarding admissions, courses, or campus information today?"}
    ]

# Render Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Processing
if user_prompt := st.chat_input("Ask anything about UoBS (e.g., What programs are offered?)..."):
    
    # Render User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Perform Retrieval
    with st.spinner("Searching UoBS knowledge base..."):
        retrieved_context = retrieve_context(user_prompt, db) if db else "No local database available."

    # Construct RAG Prompt
    system_prompt = f"""You are an official AI Assistant for the University of Baltistan, Skardu (UoBS). 
Your goal is to provide accurate, polite, and clear information to students and visitors using the provided Context.

Context from UoBS Records:
------------------
{retrieved_context}
------------------

Guidelines:
- Answer the query using the context above whenever possible.
- If the exact answer isn't in the context, use general knowledge about Pakistani universities with a courteous disclamier to verify on official portal (uobs.edu.pk).
- Be concise and structure long points with bullet points.
"""

    # Query Groq API via standard Chat Completion
    with st.chat_message("assistant"):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=MODEL_NAME,
                temperature=0.2,
                max_tokens=1024,
            )
            
            response_text = chat_completion.choices[0].message.content
            st.markdown(response_text)
            
            # Save Assistant Response
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            error_msg = f"An error occurred while contacting Groq API: {str(e)}"
            st.error(error_msg)
