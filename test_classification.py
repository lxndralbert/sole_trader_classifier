import unittest
import pandas as pd
from sole_trader_classifier import SoleTraderClassifier


class TestSoleTraderClassifier(unittest.TestCase):
    def test_classification(self):
        # Load the test data
        test_data = pd.read_csv("examples/customer_data.csv")
        expected_results = test_data["is_sole_trader"]

        # Create the classifier
        classifier = SoleTraderClassifier()

        # Make predictions and compare with expected results
        predicted_results = classifier.predict(test_data)
        self.assertEqual(len(predicted_results), len(expected_results))
        for i in range(len(predicted_results)):
            self.assertEqual(predicted_results[i], expected_results[i])
