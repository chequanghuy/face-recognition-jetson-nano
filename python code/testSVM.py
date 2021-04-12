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
# def get_embedding(model, face_pixels):
# 	# scale pixel values
# 	face_pixels = face_pixels.astype('float32')
# 	# standardize pixel values across channels (global)
# 	mean, std = face_pixels.mean(), face_pixels.std()
# 	face_pixels = (face_pixels - mean) / std
# 	# transform face into one sample
# 	samples = expand_dims(face_pixels, axis=0)
# 	# make prediction to get embedding
# 	yhat = model.predict(samples)
# 	return yhat[0]
input_file_path = 'anas.jpg'
face = Image.open(input_file_path)
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

samples = expand_dims(yhat[0], axis=0)
yhat_class = model.predict(samples)
# yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
print(class_index)
# class_probability = yhat_prob[0,class_index] * 100
# predict_names = out_encoder.inverse_transform(yhat_class)
# print('Predicted:,' class_index)