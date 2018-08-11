import os
import shutil
import json

import nbformat as nbf

def export_project(project_path='project'):
    with open('config.json') as jfile:
        config = json.load(jfile)

    # Create project dir (remove if exists)
    try:
        os.makedirs(project_path)
    except FileExistsError:
        print("Removing old project at {}".format(os.path.abspath(project_path)))
        shutil.rmtree(project_path)
        os.makedirs(project_path)

    # Copy entry
    shutil.copyfile(config['files']['entry'] + '.py', os.path.join(project_path, config['files']['entry'] + '.py'))

    # Copy jupyter notebook
    __write_ipynb(os.path.join(project_path, 'launcher.ipynb'), config)

    print("Exported project at {}".format(os.path.abspath(project_path)))

def __write_ipynb(notebook_path, config):
    nb = nbf.v4.new_notebook()

    cells = []
    cells.append(('code', """\
import os

import boto3
import sagemaker
from sagemaker.mxnet import MXNet
from sagemaker import get_execution_role
from mxnet import gluon

sagemaker_session = sagemaker.Session()

role = get_execution_role()"""))

    cells.append(('markdown', "# Dataset"))

    with open('sagemaker.ipynb') as jfile:
        note = json.load(jfile)

    cells.append(('code', note['cells'][3]['source']))

    cells.append(('markdown', "# Define model"))

    code = """\
model = MXNet("{entry}.py",
              role=role,
              train_instance_count=1,
              train_instance_type="ml.c4.xlarge",
              hyperparameters={hp})"""
    cells.append(('code', code.format(
        entry = config["files"]['entry'],
        hp = json.dumps(config['hyperparameters'], indent=16)
    )))

    cells.append(('markdown', "# Train"))

    cells.append(('code', "model.fit(inputs)"))

    cells.append(('markdown', "# Deploy"))

    cells.append(('code', "model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')"))


    for cel_type, cell_content in cells:
        if cel_type == 'code':
            nb['cells'].append( nbf.v4.new_code_cell(cell_content) )
        else:
            nb['cells'].append( nbf.v4.new_markdown_cell(cell_content) )

    nbf.write(nb, notebook_path)
