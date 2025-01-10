import unittest
from utils import predict_tags

class TestPredictTags(unittest.TestCase):
    def test_prediction(self):
        input_data = "I have a problem with a pandas dataframe and pandas series"
        expected_tags = set(["python", "pandas"])
        result = set(predict_tags(input_data)[0])
        self.assertEqual(result, expected_tags)

if __name__ == "__main__":
    unittest.main()