import mxnet as mx

def train(
        hyperparameters=None,
        input_data_config=None,
        channel_input_dirs=None,
        output_data_dir=None,
        model_dir=None,
        num_gpus=None,
        num_cpus=None,
        hosts=None,
        current_host=None,
        **kwargs):
    pass

def save(model, model_dir):
    pass

def model_fn(model_dir):
    pass

def transform_fn(input_data="",
                    content_type="text/json",
                    accept="text/json"):
    pass

# def input_fn(input_data, content_type):
#     return mx.nd.array([1, 2, 3])

# def predict_fn(block, array):
#     return mx.nd.array([1, 2, 3])

# def output_fn(ndarray, accept):
#     return ("obejct", "string")
