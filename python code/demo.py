import numpy as np
import cv2
import time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap=cv2.VideoCapture('video.avi')
start =2.0
end = 1.0
import os
import random
import numpy as np
TRT_ENGINE_PATH = '/home/huycq/tf2trt_with_onnx/facenet_engine.plan'
import inference as inf
import engine as eng
import tensorrt as trt
# import cv2
import numpy as np
import time
from PIL import Image
from numpy import asarray
from numpy import expand_dims

# from onnx_to_trt import create_engine
from joblib import dump, load
model = load('svm.joblib') 
# create_engine(ONNX_FILE_PATH, TRT_ENGINE_PATH)
TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
trt_runtime = trt.Runtime(TRT_LOGGER)

engine = eng.load_engine(trt_runtime,TRT_ENGINE_PATH )
print('Engine loaded successfully...')
while(cap.isOpened()):
    end=time.time()
    print(1./(end-start))
    start=time.time()
    ret, img = cap.read()
    if ret == True:
        # img = cv2.imread('abc.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        image = img.resize((160, 160))
        face_pixels = asarray(image)
        #  = face_array
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        # engine = eng.load_engine(trt_runtime, engine_path)
        h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
        yhat = inf.do_inference(engine, samples, h_input, d_input, h_output, d_output, stream, 1, 160, 160)

        samples = expand_dims(yhat[0], axis=0)
        yhat_class = model.predict(samples)
        # yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        print(class_index)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('show', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break

    
    # cv2.waitKey(0)
cv2.destroyAllWindows()