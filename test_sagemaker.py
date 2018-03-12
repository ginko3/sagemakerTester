import unittest
import os
import shutil
import json
import inspect

import mxnet as mx

import model_mnist as model
import model_mnist_io as io

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
        num_gpus = 0
        num_cpus = 1
        hosts = ['cpu']
        kwargs = {'key': 'value'}
        
        self.model = model.train(hyperparameters=hyperparameters,
                                    channel_input_dirs=self.channel_input_dirs,
                                    output_data_dir=self.output_data_dir,
                                    model_dir=self.model_dir,
                                    num_gpus=num_gpus,
                                    num_cpus=num_cpus,
                                    hosts=hosts,
                                    **kwargs)
    
    @classmethod
    def tearDownClass(self):
        shutil.rmtree('bucket')
        pass
    
    # ---------------------------------------------------------------------------- #
    # Test signatures                                                           #
    # ---------------------------------------------------------------------------- #
      
    def test_1_signature(self):
        if 'save' in dir(model):
            self.assertEqual(inspect.getargspec(model.save).args, ["model", "model_dir"])
        
        if 'model_fn' in dir(model):
             self.assertEqual(inspect.getargspec(model.model_fn).args, ["model_dir"])
        if 'transform_fn' in dir(model):
            self.assertEqual(inspect.getargspec(model.transform_fn).args, ["model", "input_data", "content_type", "accept"])
            
        if 'input_fn' in dir(model):
            self.assertEqual(inspect.getargspec(model.input_fn).args, ["input_data", "content_type"])
        if 'predict_fn' in dir(model):
            self.assertEqual(inspect.getargspec(model.predict_fn).args, ["block", "array"])
        if 'output_fn' in dir(model):
            self.assertEqual(inspect.getargspec(model.output_fn).args, ["ndarray", "accept"])
     
     
    # ---------------------------------------------------------------------------- #
    # Training functions                                                           #
    # ---------------------------------------------------------------------------- #

    def test_2_save(self):
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

    def test_3_transform_fn(self):
        """
        [Optional]

        Test transform_fn function.
        """
        if 'transform_fn' in dir(model):
            self.assertTrue('model_fn' in dir(model), 'Currently requires model_fn to test transform_fn')
            net = model.model_fn(self.model_dir)
            response_body, accept = model.transform_fn(model=net,
                                                        input_data=io.input_data,
                                                        content_type=io.content_type,
                                                        accept=io.accept)
            self.assertEqual(response_body, io.response_body)
            self.assertEqual(accept, io.accept)
            

    # ---------------------------------------------------------------------------- #
    # Request handlers for Gluon models                                            #
    # ---------------------------------------------------------------------------- #

    def test_4_input_predict_output_fn(self):
        """
        [Optional]

        Test input_fn function.
        """
        # if 'input_fn' in dir(model):
        #     self.assertTrue('model_fn' in dir(model), 'Currently requires model_fn to test transform_fn')
        #     net = model.model_fn(self.model_dir)
