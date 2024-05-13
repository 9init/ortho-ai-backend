import os
import cv2
import io
import torch
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from torchvision import transforms
from model_classes import SPACING # Required fir Spacing model

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get iamge edges
def get_edge_triangle_blur(image):
    
    # if pill image convert to cv2 image
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(image)
    
    # convert to cv2 image in grayscale
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    
    # Apply Canny edge detection with triangle threshold
    edge_triangle_blur = cv2.Canny(img_blur, 20, 30)
    
    return edge_triangle_blur

common_transformer = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.RandomApply([
        transforms.RandomAffine(0),
        transforms.ColorJitter(),
        transforms.GaussianBlur(5),
    ]),
])

description = {
    "lipline": [
        "the smile may considered balanced and aesthetically pleasing and some doctor may recommend lip augmentation to increase the vertical distance between the upper lip and the upper incisal edges of the teeth.",
        "the smile is considered balanced and aesthetically pleasing.",
        "the smile is considered gummy and may require lip repositioning surgery to reduce the amount of gum tissue that is exposed when smiling."
    ],
    "smile_arc": [
        "the smile is considered unattractive and may require orthodontic treatment to correct.",
        "the smile is considered unattractive and may require orthodontic treatment to correct.",
        "the smile is considered balanced and aesthetically pleasing."
    ],
    "buccal": [
        "the smile is considered balanced and aesthetically pleasing.",
        "the smile is considered balanced and aesthetically pleasing.",
        "the smile is considered unattractive and may require orthodontic treatment to correct."
    ],
    "spacing": [
        "the smile is considered balanced and aesthetically pleasing.",
        "the smile is considered unattractive and may require orthodontic treatment or dental bonding to correct."
    ],
    "tooth_decay": [
        "the smile is considered unattractive and may require dental treatment to correct.",
        "the smile is considered balanced and aesthetically pleasing."
    ]
}

def load_models():
    lipline_model = {
        "id": "lipline",
        "title": "Lipline",
        "description": "The curve of the upper lip when smiling.",
        "model": torch.load('models/libline.pth', map_location=device),
        "labels": ["Low", "Medium", "High"],
        "transform": common_transformer,
        "edge": True
    }
    smile_arc_model = {
        "id": "smile_arc",
        "title": "Smile Arc",
        "description": "The curve of the upper incisal edges of the teeth.",
        "model": torch.load('models/smile_arc.pth', map_location=device),
        "labels": ['Flat', 'Reverse', 'Parallel'],
        "edge": False,
        "transform": common_transformer
    }
    buccal_model = {
        "id": "buccal",
        "title": "Buccal Corridor",
        "description": "The dark space between the corners of the mouth and the buccal surfaces of the teeth.",
        "model": torch.load('models/buccal.pth', map_location=device),
        "labels": ['Narrow', 'Medium', 'High'],
        "edge": False,
        "transform": common_transformer
    }
    spacing_model = {
        "id": "spacing",
        "title": "Spacing",
        "description": "The space between the teeth.",
        "model": torch.load('models/spacing.pth', map_location=device),
        "labels": ['No Spacing', 'Spacing'],
        "edge": False,
        "transform": common_transformer
    }
    tooth_decay_model = {
        "id": "tooth_decay",
        "title": "Tooth Decay",
        "description": "The presence of cavities in the teeth.",
        "model": torch.load('models/tooth-decay.pth', map_location=device),
        "labels": ['Decay', 'No Decay'],
        "edge": False,
        "transform": common_transformer
    }
    
    return [lipline_model, smile_arc_model, buccal_model, spacing_model, tooth_decay_model]


    

models = load_models()

def augument_image(image):
    # Placeholder for image augmentation
    return image

def detect_faces(image):
    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    return boxes, probs, landmarks

def crop_mouth(image, landmarks):
    if landmarks is not None:
        margin = 60
        mouth_landmarks = landmarks[0][3:5]
        mouth_box = [
            min(mouth_landmarks[:,0]) - margin,
            min(mouth_landmarks[:,1]) - margin,
            max(mouth_landmarks[:,0]) + margin,
            max(mouth_landmarks[:,1]) + margin
        ]
        mouth_crop = image.crop((mouth_box[0], mouth_box[1], mouth_box[2], mouth_box[3])).convert('RGB')
        return mouth_crop
    return None

def predict_from_multiple_models(mouth_crop, models):
    predictions = []
    for model in models:
        model['model'].eval()
        aug_img = model['transform'](mouth_crop)

        if model['edge']:
            cv2_img = get_edge_triangle_blur(aug_img)
            aug_img = Image.fromarray(cv2_img).convert('RGB')

        if isinstance(aug_img, Image.Image):
            aug_img = transforms.ToTensor()(aug_img)

        with torch.no_grad():
            image_tensor = aug_img.unsqueeze(0).to(device)
            outputs = model['model'](image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        predictions.append({
            "title": model['title'],
            "description":  description[model['id']][torch.argmax(probabilities).item()],
            "labels": model['labels'],
            "classes": probabilities.detach().cpu().numpy().tolist()[0],
            "predictedIndex": torch.argmax(probabilities).item()
        })
    return predictions

@app.route('/scan', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join('uploads', filename))
    
    # Process the image
    image_path = f'uploads/{filename}'
    image = Image.open(image_path).convert('RGB')
    boxes, probs, landmarks = detect_faces(image)
    
    if boxes is not None:
        mouth_crop = crop_mouth(image, landmarks)
        
        if mouth_crop is not None:
            predictions = predict_from_multiple_models(mouth_crop, models)
            # Return the result

            img_byte_arr = io.BytesIO()
            mouth_crop.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            base64_img = base64.b64encode(img_byte_arr).decode('utf-8')
            return jsonify({
                "image": base64_img,
                "predictions": predictions
            }), 200
    
    

    return jsonify({"error": "No faces detected"}), 400

if __name__ == '__main__':
    app.run(debug=True)
