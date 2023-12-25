import json
import base64
import requests
import argparse

def make_api_call(image_file_path, furniture_labels):
    try:
        # Set the API endpoint
        api_url = "http://127.0.0.1:5000/detect-furniture"

        # Prepare the payload
        payload = {"furniture_labels": ",".join(furniture_labels)}

        # Make the API call
        with open(image_file_path, "rb") as file:
            files = {"file": (image_file_path, file)}
            response = requests.post(api_url, files=files, data=payload)

        # Parse the JSON response
        response_data = json.loads(response.text)

        print("bounding_boxes -> ", response_data.get('bounding_boxes'))
        print("labels -> ", response_data.get('labels'))
        base64_image_data = response_data.get('base64_masked_image')
        if base64_image_data:
            decoded_image_bytes = base64.b64decode(base64_image_data)

            with open('decoded_image.jpg', 'wb') as f:
                f.write(decoded_image_bytes)

    except Exception as e:
        print(f"Error making API call: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make an API call to detect furniture.")
    parser.add_argument("image_file", help="Path to the image file")
    parser.add_argument("-l", "--labels", nargs="+", help="List of furniture labels (optional)")

    args = parser.parse_args()

    image_file_path = args.image_file
    furniture_labels = args.labels if args.labels else []

    make_api_call(image_file_path, furniture_labels)

