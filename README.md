# FurnitureDetect Computer Vision API
An API endpoint that integrates the YOLOv8 model to identify and generate masks for furniture items in a submitted image.

## Setup

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

## Testing

### Unit Tests

1. **Set Up Flask App for Testing:**

   Before running tests, set Flask into testing mode. Open `app.py` and add:
   app.config['TESTING'] = True

2. **Run Unit Tests:**
   
    ```bash
    python -m unittest test_app.py
    ```

## Usage

#### API Endpoint

- **URL:** `http://127.0.0.1:5000/detect-furniture`
- **Method:** POST
- **Parameters:**
  - `file`: Image file to be uploaded (Mandatory)
  - `furniture_labels`: Comma-separated list of furniture labels (Optional)

#### Example Using Curl

```bash
curl -X POST -F file=@sample.jpg -F "furniture_labels=chair,couch" http://127.0.0.1:5000/detect-furniture -o response.json
```

#### Example Using Python Script

```bash
python test_script.py sample.jpg -l chair,couch
```

### Testing

1. **Run the Flask App:**

   Make sure the Flask app is running:

   ```bash
   python app.py
   ```

2. **Execute Test Script:**

   Run the provided test script to make an API call and save the response:

   ```bash
   python test_script.py sample.jpg -l chair,couch
   ```

3. **Check Output:**

   - The API response will be printed to the console.
   - The base64-encoded masked image will be saved as 'masked_image.jpg'.

### Additional Notes

- The app uses YOLOv8 for object detection and combines overlapping boxes for precise masking.
- Furniture labels are optional; if not provided, the default labels are 'chair', 'couch', 'bed', and 'dining table'.
- For detailed logs, refer to 'app.log' and 'utils.log'.
- Adjust file paths, configurations, and error handling based on deployment needs.
