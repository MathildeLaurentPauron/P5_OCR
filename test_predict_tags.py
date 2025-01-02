import unittest
from utils import predict_tags

class TestPredictTags(unittest.TestCase):
    def test_prediction(self):
        input_data = "Is it possible to convert a java code in c#"
        expected_tags = set(["java", "c#"])
        result = set(predict_tags(input_data)[0])
        self.assertEqual(result, expected_tags)

if __name__ == "__main__":
    unittest.main()