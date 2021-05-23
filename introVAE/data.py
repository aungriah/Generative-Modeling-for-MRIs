import tensorflow as tf
import numpy as np
import os




def read_npy_file(item):
    data = np.transpose(np.load(item.decode()), (0,3,1,2))[0,:,:,:]
    return data.astype(np.float32)


def create_dataset(path, batch_size, limit):
    dataset = tf.data.Dataset.list_files(path, shuffle=True) \
        .take((limit // batch_size) * batch_size) \
        .map(lambda x: tf.py_func(read_npy_file, [x], [tf.float32])) \
        .map(lambda x: x ) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(2)
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return (dataset, iterator, iterator_init_op, get_next)


'''
datasets_dir = 'datasets'
dataset = 'celebA-HQ-128x128'
data_path = os.path.join(datasets_dir, dataset)


dataset,iterator,iterator_init_op,get_next=create_dataset(os.path.join(data_path,"train/*.npy"),64,6400)
print(dataset)


a = np.load('/Users/aungriah/Downloads/celebA-HQ-128x128/train/imgHQ00005.npy')
print(a.shape)
b = np.load('datasets/brains/train/image0.npy')
print(b.shape)
'''


