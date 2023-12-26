import unittest
import os
from app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_detect_furniture_endpoint(self):
        # Assuming a sample image with a potted plant and a couch
        sample_image_path = 'uploads/sample.jpg'
        sample_image = open(sample_image_path, 'rb')

        # Send a POST request with 'plant' and 'couch' as furniture labels
        response = self.app.post('/detect-furniture', data={'file': (sample_image, 'sample.jpg'),
                                                            'furniture_labels': 'plant,couch'})
        self.assertEqual(response.status_code, 200)

        # Parse the JSON response
        response_data = response.get_json()

        # Verify the expected output
        expected_boxes = [{'xmax': 2000.0, 'xmin': 801.0, 'ymax': 1354.0, 'ymin': 539.0},
                          {'xmax': 599.0, 'xmin': 222.0, 'ymax': 1141.0, 'ymin': 769.0}]
        expected_labels = ['potted plant', 'couch']

        self.assertEqual(response_data.get('bounding_boxes'), expected_boxes)
        self.assertEqual(response_data.get('labels'), expected_labels)

if __name__ == '__main__':
    unittest.main()
