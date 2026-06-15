from typing import Tuple, List, Dict
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

class QAChainManager:
    """Manages question answering using RAG (Retrieval Augmented Generation)"""
    
    def __init__(self, vector_store, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        """
        Initialize QA chain manager
        
        Args:
            vector_store: VectorStoreManager instance with documents
            model_name: OpenAI model to use
            temperature: Creativity level (0 = deterministic, 1 = creative)
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.qa_chain = None
        self._initialize_chain()
    
    def _create_prompt(self) -> PromptTemplate:
        """Create custom prompt template for better answers"""
        template = """You are a helpful AI assistant that answers questions based on the provided context.
        
        Instructions:
        1. Use ONLY the information from the context below
        2. If you don't know the answer, say "I cannot find this information in the documents"
        3. Provide specific, detailed answers
        4. Cite relevant information from the context
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _initialize_chain(self):
        """Initialize the retrieval QA chain"""
        if not self.vector_store or not self.vector_store.vector_store:
            print("⚠️ Vector store not ready. Chain will be initialized when documents are processed.")
            return
        
        retriever = self.vector_store.vector_store.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
        prompt = self._create_prompt()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print(f"✅ QA chain initialized with {self.model_name}")
    
    def ask_question(self, question: str) -> Tuple[str, List[Dict]]:
        """
        Ask a question and get answer with sources
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, list of source documents)
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please process documents first.")
        
        try:
            with get_openai_callback() as callback:
                result = self.qa_chain({"query": question})
                answer = result['result']
                
                # Format source documents for display
                sources = []
                for i, doc in enumerate(result['source_documents'], 1):
                    sources.append({
                        "id": i,
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata
                    })
                
                # Log token usage (helpful for monitoring costs)
                print(f"📊 Token usage - Total: {callback.total_tokens}, "
                      f"Prompt: {callback.prompt_tokens}, "
                      f"Completion: {callback.completion_tokens}")
                
                return answer, sources
                
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, []
    
    def batch_ask(self, questions: List[str]) -> Dict[str, Tuple[str, List]]:
        """Answer multiple questions in batch"""
        results = {}
        for question in questions:
            results[question] = self.ask_question(question)
        return results
