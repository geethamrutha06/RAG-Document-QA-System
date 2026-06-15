from typing import List, Tuple
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class VectorStoreManager:
    """Manages vector database operations for semantic search"""
    
    def __init__(self, embedding_type: str = "openai"):
        """
        Initialize vector store manager
        
        Args:
            embedding_type: Type of embeddings to use ('openai' or 'local')
        """
        self.embedding_type = embedding_type
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        print("✅ Vector store manager initialized")
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"🔨 Creating FAISS vector store with {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print(f"✅ Vector store created successfully")
        
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents using cosine similarity"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create vector store first.")
        
        results = self.vector_store.similarity_search(query, k=k)
        print(f"🔍 Found {len(results)} relevant documents for query: '{query[:50]}...' وصلت هنا"  )
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def save_local(self, path: str = "./vector_db"):
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"💾 Vector store saved to {path}")
    
    def load_local(self, path: str = "./vector_db"):
        """Load vector store from disk"""
        import os
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings)
            print(f"📂 Vector store loaded from {path}")
            return True
        return False
