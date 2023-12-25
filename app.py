import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils import mask_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'avif', 'webp'}

# Set up logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.')[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/detect-furniture', methods=['POST'])
def detect_furniture():
    try:
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = f"{app.config['UPLOAD_FOLDER']}/{filename}"
            file.save(filepath)

            # Fetch furniture labels to be queried, ['chair', 'couch', 'bed', 'dining table'] as default
            query_labels = request.form.get('furniture_labels')
            if query_labels:
                furniture_labels=query_labels.split(',')
            else:
                furniture_labels=['chair', 'couch', 'bed', 'dining table']
            
            # Mask objects in the image based on the furniture labels provides
            base64_masked_image, combined_boxes, labels = mask_image(filepath, furniture_labels)

            # Prepare response with bounding boxes, labels and base64-encoded masked image
            response = {
                'bounding_boxes': combined_boxes,
                'labels': labels,
                'base64_masked_image': base64_masked_image
            }

            return jsonify(response)
        else:
            return jsonify({'error': 'File format not allowed'})
    except Exception as e:
        print(e)
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'Internal Server Error'})

if __name__ == '__main__':
    app.run(debug=True)
