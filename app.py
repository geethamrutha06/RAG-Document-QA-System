import streamlit as st
import os
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.qa_chain import QAChainManager

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

st.title("📚 RAG Document Q&A System")
st.markdown("*Ask questions about your documents with AI-powered answers*")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Documents")
    st.markdown("Supported formats: PDF, TXT")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT documents to create your knowledge base"
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents... This may take a moment"):
                # Create upload directory
                os.makedirs("uploaded_docs", exist_ok=True)
                
                # Save uploaded files
                for file in uploaded_files:
                    with open(f"uploaded_docs/{file.name}", "wb") as f:
                        f.write(file.getbuffer())
                    st.write(f"✅ Saved: {file.name}")
                
                # Process documents with RAG
                st.write("🔄 Loading and chunking documents...")
                processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
                docs = processor.load_directory("uploaded_docs")
                
                if docs:
                    st.write(f"📊 Loaded {len(docs)} document sections")
                    
                    chunks = processor.split_documents(docs)
                    st.write(f"✂️ Split into {len(chunks)} chunks")
                    
                    st.write("🔨 Building vector database...")
                    st.session_state.vector_store = VectorStoreManager()
                    st.session_state.vector_store.create_vector_store(chunks)
                    
                    st.write("🤖 Initializing QA system...")
                    st.session_state.qa_chain = QAChainManager(st.session_state.vector_store)
                    
                    st.session_state.documents_processed = True
                    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s) into {len(chunks)} chunks!")
                else:
                    st.error("❌ No valid documents found. Please check your files.")
    
    # Show status
    if st.session_state.documents_processed:
        st.divider()
        st.info("✅ System ready! Ask questions below.")

# Main chat interface
st.header("💬 Ask Questions About Your Documents")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📖 View Sources"):
                for i, src in enumerate(message["sources"], 1):
                    st.caption(f"Source {i}: {src[:200]}...")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    if st.session_state.documents_processed and st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing documents and generating answer..."):
                try:
                    answer, sources = st.session_state.qa_chain.ask_question(prompt)
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📖 View Sources"):
                            for i, src in enumerate(sources, 1):
                                if isinstance(src, dict):
                                    content = src.get('content', str(src))[:200]
                                else:
                                    content = str(src)[:200]
                                st.caption(f"Source {i}: {content}...")
                    
                    # Save to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [str(s)[:200] for s in sources] if sources else []
                    })
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}\n\nMake sure you have set your OpenAI API key in .env file"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        warning_msg = "⚠️ Please upload and process documents first using the sidebar!"
        st.warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})

# Footer
st.divider()
st.caption("🔒 Your documents are processed locally. No data is stored permanently.")
