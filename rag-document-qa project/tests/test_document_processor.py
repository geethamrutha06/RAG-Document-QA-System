"""
Unit tests for DocumentProcessor
Run with: python -m pytest tests/
"""

import unittest
import os
import tempfile
from utils.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        self.temp_dir = tempfile.mkdtemp()
    
    def test_initialization(self):
        """Test that processor initializes with correct values"""
        self.assertEqual(self.processor.chunk_size, 500)
        self.assertEqual(self.processor.chunk_overlap, 50)
        self.assertIsNotNone(self.processor.text_splitter)
    
    def test_load_empty_directory(self):
        """Test loading from non-existent directory"""
        docs = self.processor.load_directory("/nonexistent/path")
        self.assertEqual(docs, [])
    
    def test_load_directory_without_files(self):
        """Test loading from empty directory"""
        docs = self.processor.load_directory(self.temp_dir)
        self.assertEqual(docs, [])
    
    def test_split_empty_documents(self):
        """Test splitting empty document list"""
        chunks = self.processor.split_documents([])
        self.assertEqual(chunks, [])
    
    def test_get_stats_empty(self):
        """Test stats with empty documents"""
        stats = self.processor.get_stats([])
        self.assertEqual(stats["total_chunks"], 0)
        self.assertEqual(stats["total_chars"], 0)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
