"""
Unit tests for VectorStoreManager
"""

import unittest
from unittest.mock import Mock, patch
from utils.vector_store import VectorStoreManager

class TestVectorStoreManager(unittest.TestCase):
    """Test cases for VectorStoreManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = VectorStoreManager()
    
    def test_initialization(self):
        """Test that manager initializes correctly"""
        self.assertIsNotNone(self.manager)
        self.assertIsNone(self.manager.vector_store)
    
    @patch('utils.vector_store.OpenAIEmbeddings')
    def test_embeddings_initialized(self, mock_embeddings):
        """Test that embeddings are initialized"""
        mock_embeddings.assert_called()
    
    def test_similarity_search_without_store(self):
        """Test similarity search when no vector store exists"""
        with self.assertRaises(ValueError):
            self.manager.similarity_search("test query")
    
    def test_create_vector_store_empty_documents(self):
        """Test creating vector store with no documents"""
        with self.assertRaises(ValueError):
            self.manager.create_vector_store([])

if __name__ == '__main__':
    unittest.main()
