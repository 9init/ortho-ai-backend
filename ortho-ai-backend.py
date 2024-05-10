from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
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
])

def load_models():
    lipline_model = {
        "title": "Lipline",
        "description": "This model predicts the lip line",
        "model": torch.load('models/libline.pth', map_location=device),
        "classes": ["low", "medium", "high"],
        "transform": common_transformer,
        "edge": True
    }
    smile_arc_model = {
        "title": "Smile Arc",
        "description": "This model predicts the smile arc",
        "model": torch.load('models/smile_arc.pth'),
        "classes": ['Flat', 'Reverse', 'Parallel'],
        "edge": False,
        "transform": common_transformer
    }
    buccal_model = {
        "title": "Buccal Corridor",
        "description": "This model predicts the buccal corridor",
        "model": torch.load('models/buccal.pth'),
        "classes": ['Narrow', 'Medium', 'High'],
        "edge": False,
        "transform": common_transformer
    }
    spacing_model = {
        "title": "Spacing",
        "description": "This model predicts the spacing between teeth",
        "model": torch.load('models/spacing.pth'),
        "classes": ['No Spacing', 'Spacing'],
        "edge": False,
        "transform": common_transformer
    }
    
    return [lipline_model, smile_arc_model, buccal_model, spacing_model]

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
            "description": model['description'],
            "classes": model['classes'],
            "prediction": probabilities.detach().cpu().numpy().tolist()[0],
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
            print(predictions)
            # Return the result
            return jsonify(predictions), 200
    
    

    return jsonify({"error": "No faces detected"}), 400

if __name__ == '__main__':
    app.run(debug=True)
