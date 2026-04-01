import unittest
from codes import app, validate_input, predict_news

class FakeNewsDetectionTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_validate_input(self):
        # Test valid input
        self.assertEqual(validate_input("This is a valid news article."), "This is a valid news article.")
        
        # Test empty input
        with self.assertRaises(ValueError):
            validate_input("")

    def test_predict_news(self):
        # Test prediction with a sample news article
        prediction, true_prob, fake_prob = predict_news("This is a sample news article.", return_prob=True)
        self.assertIn(prediction, ["Fake News", "True News"])

    def test_predict_api_success(self):
        # Test POST request to the predict route
        response = self.app.post('/predict', json={'news': 'This is a sample news article.'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'prediction', response.data)

    def test_predict_api_empty(self):
        # Test POST request with empty news article
        response = self.app.post('/predict', json={'news': ''})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Input cannot be empty', response.data)

if __name__ == '__main__':
    unittest.main()
