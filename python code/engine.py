import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
def build_engine(onnx_path, shape = [1,160,160,3]):
    net_w, net_h = 160,160
    MAX_BATCH_SIZE=1
    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file. 
        shape : Shape of the input of the ONNX file. 
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = MAX_BATCH_SIZE
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            '000_net',                          # input tensor name
            (MAX_BATCH_SIZE, 3, net_h, net_w),  # min shape
            (MAX_BATCH_SIZE, 3, net_h, net_w),  # opt shape
            (MAX_BATCH_SIZE, 3, net_h, net_w))  # max shape
        config.add_optimization_profile(profile)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        return engine

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)
def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine
