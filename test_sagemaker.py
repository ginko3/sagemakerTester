import unittest
import os

import mxnet as mx

import example_model as model

class MXNetModelTest(unittest.TestCase):
    """Testcase to check a mxnet model before handing it to sagemaker."""

    def setUp(self):
        self.bucket_path = "bucket"
        if not os.path.exists(self.bucket_path):
            os.makedirs(self.bucket_path)

        self.channel_input_dirs = {'train': os.path.join(self.bucket_path, 'dataset/train'),
                             'eval': os.path.join(self.bucket_path, 'dataset/eval')}
        self.output_data_dir = os.path.join(self.bucket_path, 'data')
        self.model_dir = os.path.join(self.bucket_path, 'model')


    # ---------------------------------------------------------------------------- #
    # Training functions                                                           #
    # ---------------------------------------------------------------------------- #

    def test_train(self):
        """
        [Required]

        Test required train function.
        """
        hyperparameters = {'learningRate': '0.01'}
        num_gpus = 1
        num_cpus = 0
        kwargs = {'key': 'value'}

        model.train(hyperparameters=hyperparameters,
                    channel_input_dirs=self.channel_input_dirs,
                    output_data_dir=self.output_data_dir,
                    model_dir=self.model_dir,
                    num_gpus=num_gpus,
                    num_cpus=num_cpus,
                    **kwargs)


    def test_save(self):
        """
        [Optional]

        Test model_fn function.
        """
        if 'save' in dir(model):
            model.save(model="model", model_dir=self.model_dir)

    # ---------------------------------------------------------------------------- #
    # Hosting functions                                                            #
    # ---------------------------------------------------------------------------- #

    def test_model_fn(self):
        """
        [Optional]

        Test model_fn function.
        """
        if 'model_fn' in dir(model):
            model.model_fn(self.model_dir)

    def test_transform_fn(self):
        """
        [Optional]

        Test transform_fn function.
        """
        if 'transform_fn' in dir(model):
            model.transform_fn(input_data="",
                                content_type="text/json",
                                accept="text/json"
                                )


    # ---------------------------------------------------------------------------- #
    # Request handlers for Gluon models                                            #
    # ---------------------------------------------------------------------------- #

    def test_input_fn(self):
        """
        [Optional]

        Test input_fn function.
        """
        if 'input_fn' in dir(model):
            return_value = model.input_fn(input_data="", content_type="text/json")
            self.assertIsInstance(return_value, mx.nd.NDArray)

    def test_predict_fn(self):
        """
        [Optional]

        Test predict_fn function.
        """
        if 'predict_fn' in dir(model):
            return_value = model.predict_fn(block="", array=mx.nd.array([1, 2, 3]))
            self.assertIsInstance(return_value, mx.nd.NDArray)

    def test_output_fn(self):
        """
        [Optional]

        Test output_fn function.
        """
        if 'output_fn' in dir(model):
            return_value = model.output_fn(ndarray=mx.nd.array([1, 2, 3]), accept="text/json")
            self.assertIsInstance(return_value, tuple)
