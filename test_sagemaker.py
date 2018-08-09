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
        # Load parameters
        with open('config.json') as jfile:
            config = json.load(jfile)

        hyperparameters = config['hyperparameters']
        num_gpus = config['num_gpus']
        num_cpus = config['num_cpus']
        hosts = config['hosts']
        kwargs = config['kwargs']

        # Create directories
        self.bucket_path = "bucket"
        self.channel_input_dirs = {'train': config['dataset']['train'],
                             'eval': config['dataset']['eval']}
        self.output_data_dir = os.path.join(self.bucket_path, 'data')
        self.model_dir = os.path.join(self.bucket_path, 'model')

        os.makedirs(self.bucket_path)
        os.makedirs(self.model_dir)

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
            self.assertEqual(str(inspect.signature(model.save)), '(model, model_dir)')

        if 'model_fn' in dir(model):
            self.assertEqual(str(inspect.signature(model.model_fn)), '(model_dir)')
        if 'transform_fn' in dir(model):
            self.assertEqual(str(inspect.signature(model.transform_fn)), '(model, input_data, content_type, accept)')

        if 'input_fn' in dir(model):
            self.assertEqual(str(inspect.signature(model.input_fn)), '(input_data, content_type)')
        if 'predict_fn' in dir(model):
            self.assertEqual(str(inspect.signature(model.predict_fn)), '(block, array)')
        if 'output_fn' in dir(model):
            self.assertEqual(str(inspect.signature(model.output_fn)), '(ndarray, accept)')


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
            self.assertEqual(accept, io.accept)
            if 'response_body' in dir(io):
                self.assertEqual(response_body, io.response_body)
            else:
                print("\nResponse: ", response_body)


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
