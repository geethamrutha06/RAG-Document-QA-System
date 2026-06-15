"""
Unit tests for QAChainManager
"""

import unittest
from unittest.mock import Mock, patch
from utils.qa_chain import QAChainManager

class TestQAChainManager(unittest.TestCase):
    """Test cases for QAChainManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.mock_vector_store.vector_store = Mock()
        self.qa_manager = QAChainManager(self.mock_vector_store)
    
    def test_initialization(self):
        """Test that QA manager initializes correctly"""
        self.assertIsNotNone(self.qa_manager)
        self.assertEqual(self.qa_manager.model_name, "gpt-3.5-turbo")
        self.assertEqual(self.qa_manager.temperature, 0)
    
    def test_prompt_creation(self):
        """Test that custom prompt is created"""
        prompt = self.qa_manager._create_prompt()
        self.assertIsNotNone(prompt)
        self.assertIn("context", prompt.input_variables)
        self.assertIn("question", prompt.input_variables)
    
    def test_ask_question_without_chain(self):
        """Test asking question when chain not initialized"""
        self.qa_manager.qa_chain = None
        with self.assertRaises(ValueError):
            self.qa_manager.ask_question("test question")

if __name__ == '__main__':
    unittest.main()
