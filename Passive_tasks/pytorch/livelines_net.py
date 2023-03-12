import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json

import torchvision.models as models

import torch.nn as nn

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Costum_MobileNetV2(nn.Module):
    def __init__(self):
        super(Costum_MobileNetV2,self).__init__()
        self.base = models.mobilenet_v2(pretrained=True).to(device)
        self.base.classifier = nn.Linear(in_features=1280, out_features= 1)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        input = input.float()
        x = self.base(input)
        x = self.activation(x)
        return(x)

model = Costum_MobileNetV2().to(device)

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# # Load Anti-Spoofing Model graph
# json_file = open('./antispoofing_models/Fine_Tuned_nodel_Mobile_NET_V2.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_state_dict(torch.load('./antispoofing_models/Fine_Tuned_nodel_Mobile_NET_V2.pt', map_location=torch.device('cpu')))

# model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")
# video.open("http://192.168.1.101:8080/video")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

video = cv2.VideoCapture(0)
while True:
    
    ret,frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:  
        face = frame[y-5:y+h+5,x-5:x+w+5]
        resized_face = cv2.resize(face,(160,160))
        resized_face = resized_face.astype("float") / 255.0
        # resized_face = img_to_array(resized_face)
        resized_face = np.expand_dims(resized_face, axis=0)
        resized_face = torch.from_numpy(resized_face)
        # pass the face ROI through the trained liveness detector
        # model to determine if the face is "real" or "fake"
        preds = model(resized_face.permute(0,3,1,2))
        print(preds)
        if preds> 0.5:
            label = 'spoof'
            cv2.putText(frame, label, (x,y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.rectangle(frame, (x, y), (x+w,y+h),
                (0, 0, 255), 2)
        else:
            label = 'real'
            cv2.putText(frame, label, (x,y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w,y+h),
            (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        # Capture frame-by-frame

video.release()        
cv2.destroyAllWindows()