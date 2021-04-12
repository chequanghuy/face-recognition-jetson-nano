
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
# from onnx_to_trt import create_engine

# create_engine(ONNX_FILE_PATH, TRT_ENGINE_PATH)
TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
trt_runtime = trt.Runtime(TRT_LOGGER)

engine = eng.load_engine(trt_runtime,TRT_ENGINE_PATH )
print('Engine loaded successfully...')
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


filenames=os.listdir("faceClass2/")
# print(filenames)
# print(filenames)
labels=[]
paths=[]
# images=[]
# images_test=[]
# c=0
for i in filenames:
    path=os.path.join("faceClass2/",i)
    file_images=os.listdir(path)
    # print(file_images)
    for j,f in enumerate(file_images):
        image=os.path.join(path,f)       
        paths.append(image)
random.shuffle(paths)
# print(paths)
from sklearn.preprocessing import OneHotEncoder
labels = [p.split(os.path.sep)[-2] for p in paths]
labels = np.array(labels)
print(labels.shape)
# labels = labels.reshape(len(labels), 1)
# onehot_encoder = OneHotEncoder()
# labels = onehot_encoder.fit_transform(labels)
X = list()
# print(paths.size)
for filename in paths:
# load image from file
    face = Image.open(filename)
    # convert to RGB, if needed
    image = face.convert('RGB')
    # image = Image.fromarray(face)
    image = image.resize((160, 160))
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
    X.append(yhat[0])
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
# X_train = np.array(X_train)
# print(y_train.shape)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

from joblib import dump, load
dump(model, 'svm.joblib') 
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)
# score
score_train = accuracy_score(y_train, yhat_train)
score_test = accuracy_score(y_test, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

