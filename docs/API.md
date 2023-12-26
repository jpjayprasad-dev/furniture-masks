## Documentation

### API Endpoint: `/detect-furniture`

This API endpoint receives an image file and optional furniture labels, runs object detection using YOLOv8, and returns information about the detected furniture items.

### Setup Instructions:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jpjayprasad-dev/furniture-masks
   cd furniture-masks
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask App:**
   ```bash
   python app.py
   ```

   The app will run on `http://127.0.0.1:5000/`. Ensure that the required models and dependencies are installed.

### Usage Instructions:

#### API Endpoint:

- **URL:** `http://127.0.0.1:5000/detect-furniture`
- **Method:** POST
- **Parameters:**
  - `file`: Image file to be uploaded (Mandatory)
  - `furniture_labels`: Comma-separated list of furniture labels (Optional)

#### Example Using Curl:

```bash
curl -X POST -F file=@sample.jpg -F "furniture_labels=chair,couch" http://127.0.0.1:5000/detect-furniture -o response.json
```

#### Example Using Python Script:

```bash
python test_script.py sample.jpg -l chair,couch
```

### Output:

The API response will include the following information:

- `bounding_boxes`: Detected bounding boxes for furniture items.
- `labels`: Labels corresponding to the detected furniture items.
- `base64_masked_image`: Base64-encoded masked image with highlighted furniture items.

### Additional Notes:

- The app uses YOLOv8 for object detection and combines overlapping boxes for precise masking.
- Furniture labels are optional; if not provided, the default labels are 'chair', 'couch', 'bed', and 'dining table'.

## Approach Explanation:

1. **Flask App (`app.py`):**
   - Defines a Flask app with an endpoint `/detect-furniture`.
   - Accepts image files and optional furniture labels as input.
   - Uses YOLOv8 for object detection and the `utils.mask_image` function for further processing.
   - Returns bounding boxes, labels, and a base64-encoded masked image in the API response.

2. **Utility Functions (`utils.py`):**
   - Implements functions for object detection, masking, combining overlapping boxes, and generating label embeddings.
   - Employs the SentenceTransformer library to generate text embeddings for the queried furniture labels. The generated embeddings are then utilized in conjunction with AnnoyIndex to match them with the object names retrieved from the image using the YOLOv8 model.

3. **Test Script (`test_script.py`):**
   - Provides an example of making an API call using a Python script.
   - Takes image file path and optional furniture labels as command-line arguments.
   - Parses the API response and saves the base64-encoded masked image locally.

### Note:

- Ensure the app is running (`python app.py`) before making API calls.
- Check the `requirements.txt` file for the required Python packages.
- For detailed logs, refer to 'app.log' and 'utils.log'.
- Adjust file paths, configurations, and error handling based on deployment needs.

## References:

- YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- SentenceTransformer: [https://www.sbert.net/](https://www.sbert.net/)
- AnnoyIndex: [https://github.com/spotify/annoy](https://github.com/spotify/annoy)