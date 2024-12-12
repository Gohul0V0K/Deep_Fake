import json
import glob
import numpy as np
import cv2
import os
import face_recognition
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
from tqdm.autonotebook import tqdm
from torch.cuda.amp import autocast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Imported")

# Preprocessing part
def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = True
    frame_count = 0
    while success:
        success, image = vidObj.read()
        if success:
            frame_count += 1
            print(f"Extracted frame {frame_count}")
            yield image

def create_face_videos_and_predict(path_list, out_dir, model, transform):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results = []

    for path in tqdm(path_list, desc="Processing videos"):
        out_path = os.path.join(out_dir, os.path.basename(path))
        print(f"Processing video: {path}")
        
        frames = []

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))
        for idx, frame in enumerate(frame_extract(path)):
            try:
                print("Running face detection on individual frame...")
                face_locations = face_recognition.face_locations(frame)  
                print(f"Faces detected in frame {idx + 1}: {face_locations}")
                
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    cropped_face = frame[top:bottom, left:right]
                    frames.append(transform(cv2.resize(cropped_face, (112, 112))))
                    out.write(cv2.resize(cropped_face, (112, 112)))


                    # Predict for each face individually
                    # prediction = predict(model, processed_face)
                    # results.append((path, prediction))
                    # print(f"Prediction for frame {idx + 1}: {prediction}")

            except Exception as e:
                print(f"Error during processing frame {idx + 1}: {e}")

        out.release()
        print(f"Finished processing video: {path}")

        if len(frames) > 0:
            frames = torch.stack(frames)
            frames = frames.unsqueeze(0)
            prediction = predict(model, frames)
            results.append((path, prediction))
            print(f"Prediction for {path}: {prediction}")

    return results

# Prediction part
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        print("\nModel initialized\n")

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))

def im_convert(tensor):
    print("\nImage conversion started\n")
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png', image * 255)
    return image

def predict(model, img):
    img = img.to('cuda').half()  # Convert input to float16 and move to GPU
    model = model.half()  # Ensure model is also in float16
    with torch.no_grad():
        with autocast():
            fmap, logits = model(img)
            logits = sm(logits)
            _, prediction = torch.max(logits, 1)
            confidence = logits[:, int(prediction.item())].item() * 100
            print('Confidence of prediction:', confidence)
            
            return [int(prediction.item()), confidence]

# Loading and predicting
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

path_to_videos = glob.glob('qwert/*.mp4')
model = Model(2).cuda()
path_to_model = 'train_3_result/checkpoint3.pt'
model.load_state_dict(torch.load(path_to_model))
model.eval()

results = create_face_videos_and_predict(path_to_videos, "output", model, train_transforms)

print("Prediction Results:")
for path, (label, confidence) in results:
    print(f"{path} - {'REAL' if label == 1 else 'FAKE'} with {confidence:.2f}% confidence")
