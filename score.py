import os
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import json
from PIL import Image
import requests
from io import BytesIO
import time
import datetime
import torch.nn.functional as F
import numpy as np
import base64


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_ch)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class ResEmoteNet(nn.Module):
    def __init__(self):
        super(ResEmoteNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(256)
        
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 7)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.se(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

def init():
    global model, labels, face_classifier, device, preprocess
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pth')
    model = ResEmoteNet()
    checkpoint = torch.load(model_path, lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Define labels
    labels = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
    
    # Initialize face classifier
    try:
        # Try to load from model directory first
        face_cascade_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'haarcascade_frontalface_default.xml')
        if os.path.exists(face_cascade_path):
            face_classifier = cv2.CascadeClassifier(face_cascade_path)
        else:
            # Use OpenCV's built-in cascades
            face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        import pkg_resources
        opencv_data_path = pkg_resources.resource_filename('cv2', 'data')
        face_classifier = cv2.CascadeClassifier(os.path.join(opencv_data_path, 'haarcascade_frontalface_default.xml'))
    
    # Define preprocessing transform
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def detect_faces(image):
    """
    Detect faces in an image using OpenCV
    """
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image)
    # Convert RGB to BGR (OpenCV uses BGR)
    if open_cv_image.shape[2] == 3:
        open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    return open_cv_image, faces

def run(input_data):
    data = json.loads(input_data)['image']
    prev_time = time.time()

    if data.startswith('http'):
        response = requests.get(data)
        input_image = Image.open(BytesIO(response.content))
    else:
        if data.startswith('data:image'):
            data = data.split(',')[1]
        image_data = base64.b64decode(data)
        input_image = Image.open(BytesIO(image_data))
    
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    
    cv_image, faces = detect_faces(input_image)
    
    if len(faces) == 0:
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_batch)
        
        probabilities = F.softmax(output[0], dim=0)
        index = output.data.cpu().numpy().argmax()
        probability = probabilities.data.cpu().numpy().max()
    else:
        # Get the first face
        x, y, w, h = faces[0]
        
        # Crop the face
        face_img = cv_image[y:y+h, x:x+w]
        # Convert back to PIL Image
        face_pil = Image.fromarray(face_img[:, :, ::-1])  # Convert BGR back to RGB
        
        # Preprocess and run inference
        input_tensor = preprocess(face_pil)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_batch)
        
        probabilities = F.softmax(output[0], dim=0)
        index = output.data.cpu().numpy().argmax()
        probability = probabilities.data.cpu().numpy().max()
    
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    
    # Format output exactly like the original
    predictions = {}
    predictions[labels[index]] = str(round(float(probability)*100, 2))
    
    result = {
        'time': str(inference_time.total_seconds()),
        'prediction': labels[index],
        'scores': predictions
    }

    return result