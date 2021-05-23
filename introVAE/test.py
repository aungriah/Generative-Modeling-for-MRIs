import numpy as np
import tensorflow as tf
import keras, keras.backend as K

from keras.layers import Input
from keras.models import Model

import os, sys, time
from collections import OrderedDict

import model, losses, utils, data

####################################################################################################
####################################################################################################
####################################### Predefined Arguments #######################################

# datasets
args_datasets_dir = 'datasets'
args_dataset = 'brains'
args_color = False
args_train_size = 20000
args_test_size = 2484
args_shape = 128,128

# model hyperparameters
args_m = 110
args_alpha = 0.25
args_beta = 2.5

# training
args_lr = 0.0002
args_batch_size = 8
args_nb_epoch = 200
args_verbose = 1

# architecture
args_sampling = True
args_sampling_std = -1.0
args_latent_dim = 256
args_base_filter_num = 16
args_resnet_wideness = 1

# encoder
args_encoder_use_bn = True
args_encoder_wd = 0.0

# generator
args_generator_use_bn = True
args_generator_wd = 0.0

# locations
args_prefix = 'model/pictures_inference'
args_model_path = 'trained_models/models/bestmodel'
#args_model_path = 'trained_models/best_model'


# micellaneous
args_memory_share = 0.95
args_frequency = 100
args_save_latent = True
args_latent_cloud_size = 10000

# set random seed
np.random.seed(10)
####################################################################################################
####################################################################################################
####################################################################################################


print('Keras version: ', keras.__version__)
print('Tensorflow version: ', tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args_memory_share
set_session(tf.Session(config=config))

#
# Datasets
#

def interpolate(steps,z1,z2):
    diff = (z2-z1)/steps
    z1 = np.expand_dims(z1,axis=0)
    for i in range(steps):
        z = z1 + (i+1)*diff
        z = np,expan_dims(z,axis=0)
        z1 = np.concatenate((z1,z),axis = 0)
    return z1



K.set_image_data_format('channels_first')

data_path = os.path.join(args_datasets_dir, args_dataset)

iterations = args_nb_epoch * args_train_size // args_batch_size
iterations_per_epoch = args_train_size // args_batch_size

train_dataset, train_iterator, train_iterator_init_op, train_next \
     = data.create_dataset(os.path.join(data_path, "train/*.npy"), args_batch_size, args_train_size)
test_dataset, test_iterator, test_iterator_init_op, test_next \
     = data.create_dataset(os.path.join(data_path, "test/*.npy"), args_batch_size, args_test_size)
fixed_dataset, fixed_iterator, fixed_iterator_init_op, fixed_next \
     = data.create_dataset(os.path.join(data_path, "train/*.npy"), args_batch_size, args_latent_cloud_size)

args_n_channels = 3 if args_color else 1
args_n_channels = 1
args_original_shape = (args_n_channels, ) + args_shape
print(args_original_shape)
#
# Build networks
#

encoder_layers = model.encoder_layers_introvae(args_shape, args_base_filter_num, args_encoder_use_bn)
generator_layers = model.generator_layers_introvae(args_shape, args_base_filter_num, args_generator_use_bn)

encoder_input = Input(batch_shape=[args_batch_size] + list(args_original_shape), name='encoder_input')
generator_input = Input(batch_shape=(args_batch_size, args_latent_dim), name='generator_input')

encoder_output = encoder_input
for layer in encoder_layers:
    encoder_output = layer(encoder_output)

generator_output = generator_input
for layer in generator_layers:
    generator_output = layer(generator_output)

z, z_mean, z_log_var = model.add_sampling(encoder_output, args_sampling, args_sampling_std, args_batch_size, args_latent_dim, args_encoder_wd)

encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var])
generator = Model(inputs=generator_input, outputs=generator_output)

print('Encoder')
encoder.summary()
print('Generator')
generator.summary()

xr = generator(z)
reconst_latent_input = Input(batch_shape=(args_batch_size, args_latent_dim), name='reconst_latent_input')
zr_mean, zr_log_var = encoder(generator(reconst_latent_input))
zr_mean_ng, zr_log_var_ng = encoder(tf.stop_gradient(generator(reconst_latent_input)))
xr_latent = generator(reconst_latent_input)

sampled_latent_input = Input(batch_shape=(args_batch_size, args_latent_dim), name='sampled_latent_input')
zpp_mean, zpp_log_var = encoder(generator(sampled_latent_input))
zpp_mean_ng, zpp_log_var_ng = encoder(tf.stop_gradient(generator(sampled_latent_input)))

encoder_optimizer = tf.train.AdamOptimizer(learning_rate=args_lr)
generator_optimizer = tf.train.AdamOptimizer(learning_rate=args_lr)

print('Encoder')
encoder.summary()
print('Generator')
generator.summary()

#
# Define losses
#

l_reg_z = losses.reg_loss(z_mean, z_log_var)
l_reg_zr_ng = losses.reg_loss(zr_mean_ng, zr_log_var_ng)
l_reg_zpp_ng = losses.reg_loss(zpp_mean_ng, zpp_log_var_ng)

l_ae = losses.mse_loss(encoder_input, xr, args_original_shape)
l_ae2 = losses.mse_loss(encoder_input, xr_latent, args_original_shape)

encoder_l_adv = l_reg_z + args_alpha * K.maximum(0., args_m - l_reg_zr_ng) + args_alpha * K.maximum(0., args_m - l_reg_zpp_ng)
encoder_loss = encoder_l_adv + args_beta * l_ae

l_reg_zr = losses.reg_loss(zr_mean, zr_log_var)
l_reg_zpp = losses.reg_loss(zpp_mean, zpp_log_var)

generator_l_adv = args_alpha * l_reg_zr + args_alpha * l_reg_zpp
generator_loss = generator_l_adv + args_beta * l_ae2

#
# Define training step operations
#

encoder_params = encoder.trainable_weights
generator_params = generator.trainable_weights

encoder_grads = encoder_optimizer.compute_gradients(encoder_loss, var_list=encoder_params)
encoder_apply_grads_op = encoder_optimizer.apply_gradients(encoder_grads)

generator_grads = generator_optimizer.compute_gradients(generator_loss, var_list=generator_params)
generator_apply_grads_op = generator_optimizer.apply_gradients(generator_grads)

for v in encoder_params:
    tf.summary.histogram(v.name, v)
for v in generator_params:
    tf.summary.histogram(v.name, v)
summary_op = tf.summary.merge_all()

#
# Main loop
#

print('Start session')
global_iters = 0
start_epoch = 0

with tf.Session() as session:

    init = tf.global_variables_initializer()
    session.run([init, train_iterator_init_op, test_iterator_init_op, fixed_iterator_init_op])

    #summary_writer = tf.summary.FileWriter(args_prefix+"/", graph=tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)
    #if args_model_path is not None and tf.train.checkpoint_exists(args_model_path):
    saver.restore(session, tf.train.latest_checkpoint(args_model_path))
    print('Model restored from ' + args_model_path)
    ckpt = tf.train.get_checkpoint_state(args_model_path)
    global_iters = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
    start_epoch = (global_iters * args_batch_size) // args_train_size
    print('Global iters: ', global_iters)


    x = session.run(train_next)
    print(x.shape)
    pics = np.load('/data/normal_brains.npy')
    pics = np.expand_dims(pics,axis=1)

    x = pics[:16]
    #idx = np.random.randint(0,50000,8)
    #x = train_dataset[idx]
    z_p = np.random.normal(loc=0.0, scale=1.0, size=(args_batch_size, args_latent_dim))
    z_x, x_r, x_p = session.run([z, xr, generator_output], feed_dict={encoder_input: x, generator_input: z_p})
    print(z_x.shape)
    z1 = z_x[0]
    z2 = z_x[args_batch_size-1]
    diff = (z2-z1)/15
    zz =np.expand_axisdims(z1,axis=0)
    for i in range(15):
        z_inter = z1 + (i+1)*diff
        z_inter = np.expand_dims(z_inter,axis=0)
        zz = np.concatenate((zz,z_inter),axis=0)
    interp1 = session.run(xr,feed_dict={z:zz[:8]})
    interp2 = session.run(xr,feed_dict={z:zz[8:]})
    recon = session.run(xr,feed_dict={z:z_x})

    n_x = 5
    n_y = 64 // n_x
    print('Save original images.')
    utils.plot_images(np.transpose(x, (0, 2, 3, 1)), n_x, n_y, "original_images", text=None)
    print('Save generated images.')
    utils.plot_images(np.transpose(x_p, (0, 2, 3, 1)), n_x, n_y, "sampled_images", text=None)
    print('Save reconstructed images.')
    utils.plot_images(np.transpose(recon, (0, 2, 3, 1)), n_x, n_y, "reconstructed_images", text=None)
    print('Save interpolated images.')
    utils.plot_images(np.transpose(np.concatenate((interp1,interp2),axis=0), (0, 2, 3, 1)), n_x, n_y, "interpolated_images", text=None)
