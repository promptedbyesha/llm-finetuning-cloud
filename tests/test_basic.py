import unittest
from src.preprocessing.preprocess_data import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def test_tokenize(self):
        sample = ["This is a test."]
        result = preprocess_data(sample)
        self.assertIn("input_ids", result)

if __name__ == "__main__":
    unittest.main()
