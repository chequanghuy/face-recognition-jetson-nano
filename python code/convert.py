from keras_to_pb_tf1  import keras_to_pb
from keras.models import load_model



#User defined values
#Input file path
MODEL_PATH = '/home/huycq/tf2trt_with_onnx/facenet_keras.h5'
#output files paths
PB_FILE_PATH = '/home/huycq/tf2trt_with_onnx/facenet.pb'
ONNX_FILE_PATH = '/home/huycq/tf2trt_with_onnx/facenet_onnx.onnx'
TRT_ENGINE_PATH = '/home/huycq/tf2trt_with_onnx/facenet_engine.plan'
import inference as inf
import engine as eng
import tensorrt as trt
import cv2
import numpy as np
import time
from PIL import Image
from onnx_to_trt import create_engine

create_engine(ONNX_FILE_PATH, TRT_ENGINE_PATH)
TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
trt_runtime = trt.Runtime(TRT_LOGGER)

engine = eng.load_engine(trt_runtime,TRT_ENGINE_PATH )
print('Engine loaded successfully...')
for i in range(0,5):
    t=time.time()
    face=cv2.imread('anas.png')
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)

    face_pixels = face_array
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # samples=cv2.resize(samples,(160,160))
    # # samples=[samples]
    # samples=np.array(samples.reshape(1,160,160,3))
    

    h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
    out = inf.do_inference(engine, samples, h_input, d_input, h_output, d_output, stream, 1, 160, 160)
    print(1/(time.time()-t))
