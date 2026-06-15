import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    """Handles loading and chunking of documents for RAG pipeline"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory"""
        documents = []
        
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist")
            return documents
        
        supported = ['.pdf', '.txt']
        files_loaded = 0
        
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext not in supported:
                continue
                
            try:
                if ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"✅ Loaded PDF: {file} ({len(docs)} pages)")
                    files_loaded += 1
                    
                elif ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"✅ Loaded TXT: {file}")
                    files_loaded += 1
                    
            except Exception as e:
                print(f"❌ Error loading {file}: {str(e)}")
        
        print(f"📚 Total: {len(documents)} document sections from {files_loaded} files")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval"""
        if not documents:
            print("No documents to split")
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    def get_stats(self, documents: List[Document]) -> dict:
        """Get statistics about processed documents"""
        if not documents:
            return {"total_chunks": 0, "total_chars": 0, "avg_chunk_size": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        return {
            "total_chunks": len(documents),
            "total_chars": total_chars,
            "avg_chunk_size": total_chars // len(documents) if documents else 0
        }
