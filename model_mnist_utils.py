import mxnet as mx
from mxnet import gluon

def get_train_data(data_dir, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.MNIST(data_dir, train=True, transform=__input_transformer),
        batch_size=batch_size, shuffle=True, last_batch='discard')


def get_val_data(data_dir, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.MNIST(data_dir, train=False, transform=__input_transformer),
        batch_size=batch_size, shuffle=False)

def __input_transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label
