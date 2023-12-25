import cv2
import os
import base64
import logging
import numpy as np
from shapely.geometry import box as shapely_box
import shapely
from shapely.ops import unary_union
import torch
from sentence_transformers import SentenceTransformer  # Library for text embeddings
from annoy import AnnoyIndex  # Vector database library
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
model.to('cpu')

# Set up logging
logging.basicConfig(filename='utils.log', level=logging.ERROR)

def mask_image(filepath, queries):
    try:
        # Run YOLOv8 detection
        results = model(filepath)

        # Create an AnnoyIndex with suitable dimensions
        f = 769  # Label embedding(768) + Probabability (1) + Coordinates (4)
        t = AnnoyIndex(f, 'angular')

        i = 0 # vector index
        objDict = {} # store detected objects
        for r in results:
            for box in r.boxes:
                # Retrieve box coordinates of objects
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]

                # Retrieve label of the object
                label = r.names[box.cls[0].item()]

                # Retrieve probability of object against the name
                prob = round(box.conf[0].item(), 2)

                # Generate word embedding of the label
                label_embedding = generate_label_embedding(str(label))

                # Create a combined embedding of label, probability and coordinates
                combined_embedding = np.concatenate((label_embedding, np.array([prob])))

                print("Adding label/prob/cords ", label, "/", prob, "/", cords)

                # Add the combined embedding in Annoy Index
                t.add_item(i, combined_embedding)

                if label not in objDict:
                    objDict[label] = { 'cords' : [cords], 'indexes' : [i]}
                else:
                    objDict[label]['cords'].append(cords)
                    objDict[label]['indexes'].append(i)

                i += 1

        # Build the index
        t.build(10)

        # Fetch boxes which is matching with the queries
        boxes = []
        labels = []
        for query in queries:
            # Generate embedding of the provided query
            query_embedding = generate_label_embedding(query)

            # Create combined embedding for close match
            combined_embedding = np.concatenate((query_embedding, np.array([.9])))
            nearest_indexes, distances = t.get_nns_by_vector(combined_embedding, 1, include_distances=True)

            for i in range(len(nearest_indexes)):
                nearest_index = nearest_indexes[i]
                distance = distances[i]
                print("nearest_index/distance ", nearest_index, "/", distance)
                if nearest_index != -1:
                    for name, obj in objDict.items():
                        if nearest_index in obj['indexes']:
                            cords = obj['cords']
                            if name not in labels:
                                labels.append(name)
                            for cord in cords:
                                box = {'xmin' : cord[0], 'ymin' : cord[1], 'xmax' : cord[2], 'ymax':cord[3]}
                                boxes.append(box)
                                print("cords/query", cord, "/", query)

        if boxes:
            # Detect and combine overlapping boxes
            combined_boxes = combine_overlapping_boxes(boxes)

            # Generate base64-encoded masked image for combined boxes
            base64_masked_image = generate_masks(filepath, combined_boxes)

            # Return masked image, combined boxes and names of detected objects
            return base64_masked_image, combined_boxes, labels
        else:
            return "", [], []
    except Exception as e:
        logging.error(f"An error occurred while masking image: {str(e)}")
        raise

def combine_overlapping_boxes(boxes):
    try:
        # Convert bounding boxes to shapely.geometry.box objects
        shapely_boxes = [shapely_box(box['xmin'], box['ymin'], box['xmax'], box['ymax']) for box in boxes]

        # Use unary_union to combine only overlapping boxes
        combined_boxes = unary_union(shapely_boxes)

        # Convert combined_boxes back to bounding box format
        if isinstance(combined_boxes, shapely.geometry.polygon.Polygon):
            merged_bounds = [{'xmin': combined_boxes.bounds[0], 'ymin': combined_boxes.bounds[1],
                              'xmax': combined_boxes.bounds[2], 'ymax': combined_boxes.bounds[3]}]
        elif isinstance(combined_boxes, shapely.geometry.multipolygon.MultiPolygon):
            merged_bounds = [{'xmin': poly.bounds[0], 'ymin': poly.bounds[1],
                              'xmax': poly.bounds[2], 'ymax': poly.bounds[3]} for poly in combined_boxes.geoms]
        else:
            raise ValueError("Unsupported geometry type")
        return merged_bounds
    except Exception as e:
        logging.error(f"An error occurred in combine_overlapping_boxes: {str(e)}")
        raise

def generate_masks(image_path, boxes):
    try:
        image = cv2.imread(image_path)
        masked_image = image.copy()

        for idx, box in enumerate(boxes):
            # Convert box coordinates to integers
            xmin, ymin, xmax, ymax = map(int, (box['xmin'], box['ymin'], box['xmax'], box['ymax']))
            # Generate a mask for the current box
            mask = np.zeros_like(image)
            mask[ymin:ymax, xmin:xmax] = image[ymin:ymax, xmin:xmax]
            masked_image = cv2.addWeighted(masked_image, 1, mask, -1, 0)

        # Convert the masked image to base64
        _, encoded_image = cv2.imencode('.jpg', masked_image)
        base64_image = base64.b64encode(encoded_image).decode('utf-8')

        # Save the masked image to the file system
        output_path = os.path.join('output', 'masked_image.jpg')
        cv2.imwrite(output_path, masked_image)

        return base64_image
    except Exception as e:
        logging.error(f"An error occurred in generate_masks: {str(e)}")
        raise

def generate_label_embedding(label):
    try:
        # Load the text embedding model (all-mpnet-base-v2)
        embedder = SentenceTransformer('all-mpnet-base-v2')

        # Ensure label is a string
        label_str = str(label)

        # Encode the label using the model
        with torch.no_grad():
            label_embedding = embedder.encode(label_str, convert_to_tensor=True).squeeze(0)  # Remove batch dimension

        # Return the embedding as a 1D NumPy array
        return label_embedding.cpu().numpy()
    except Exception as e:
        logging.error(f"An error occurred in generate_label_embedding: {str(e)}")
        raise


