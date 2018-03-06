import unittest
import os
import shutil
import json

import mxnet as mx

import model as model
import model_io as io

# import example_model as model

class MXNetModelTest(unittest.TestCase):
    """Testcase to check a mxnet model before handing it to sagemaker."""
    
    @classmethod
    def setUpClass(self):
        self.bucket_path = "bucket"
        self.channel_input_dirs = {'train': os.path.join(self.bucket_path, 'dataset/train'),
                             'eval': os.path.join(self.bucket_path, 'dataset/eval')}
        self.output_data_dir = os.path.join(self.bucket_path, 'data')
        self.model_dir = os.path.join(self.bucket_path, 'model')
        
        os.makedirs(self.bucket_path)
        os.makedirs(self.model_dir)
        
        hyperparameters = {'learningRate': '0.01', 'epochs': 1}
        num_gpus = 1
        num_cpus = 0
        kwargs = {'key': 'value'}

        self.model = model.train(hyperparameters=hyperparameters,
                                    channel_input_dirs=self.channel_input_dirs,
                                    output_data_dir=self.output_data_dir,
                                    model_dir=self.model_dir,
                                    num_gpus=num_gpus,
                                    num_cpus=num_cpus,
                                    **kwargs)
    
    @classmethod
    def tearDownClass(self):
        shutil.rmtree('bucket')
        pass
        
    # ---------------------------------------------------------------------------- #
    # Training functions                                                           #
    # ---------------------------------------------------------------------------- #

    def test_1_save(self):
        """
        [Optional]
        
        Test save function.
        """
        if 'save' in dir(model):
            self.assertIsNotNone(self.model, "save function defined but not used")
            model.save(model=self.model, model_dir=self.model_dir)

    # ---------------------------------------------------------------------------- #
    # Hosting functions                                                            #
    # ---------------------------------------------------------------------------- #

    def test_2_model_fn(self):
        """
        [Optional]

        Test model_fn function.
        """
        if 'model_fn' in dir(model):
            self.assertIsNotNone(self.model, "model_fn function defined but not used")
            model.model_fn(self.model_dir)


    def test_3_transform_fn(self):
        """
        [Optional]

        Test transform_fn function.
        """
        if 'transform_fn' in dir(model):
            self.assertIsNotNone(self.model, "transform_fn function defined but not used")
            response_body, accept = model.transform_fn(model=self.model,
                                                        input_data=io.input_data,
                                                        content_type=io.content_type,
                                                        accept=io.accept)
            self.assertEqual(response_body, io.response_body)
            


    # ---------------------------------------------------------------------------- #
    # Request handlers for Gluon models                                            #
    # ---------------------------------------------------------------------------- #

    def test_input_fn(self):
        """
        [Optional]

        Test input_fn function.
        """
        # if 'input_fn' in dir(model):
        #     return_value = model.input_fn(input_data="", content_type="text/json")
        #     self.assertIsInstance(return_value, mx.nd.NDArray)

    def test_predict_fn(self):
        """
        [Optional]

        Test predict_fn function.
        """
        # if 'predict_fn' in dir(model):
        #     return_value = model.predict_fn(block="", array=mx.nd.array([1, 2, 3]))
        #     self.assertIsInstance(return_value, mx.nd.NDArray)

    def test_output_fn(self):
        """
        [Optional]

        Test output_fn function.
        """
        # if 'output_fn' in dir(model):
        #     return_value = model.output_fn(ndarray=mx.nd.array([1, 2, 3]), accept="text/json")
        #     self.assertIsInstance(return_value, tuple)
