import unittest
from src.preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def test_sample(self):
        sample_data = ["Hello World!"]
        processed = preprocess_data(sample_data)
        self.assertIsInstance(processed, list)
        self.assertTrue(all(isinstance(s, str) for s in processed))

if __name__ == "__main__":
    unittest.main()
