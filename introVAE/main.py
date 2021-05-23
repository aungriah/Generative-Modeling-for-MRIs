import numpy as np
import tensorflow as tf
import keras, keras.backend as K

from keras.layers import Input
from keras.models import Model

import os, sys, time
from collections import OrderedDict

import model, losses, utils, data #, params

#
# Config
#

####################################################################################################
####################################################################################################
####################################### Predefined Arguments #######################################

# datasets
args_datasets_dir = 'datasets'
args_dataset = 'brains'
args_color = False
args_train_size = 20000
args_test_size = 22484
args_shape = 128,128
#parser.add_argument('--dataset', dest="dataset", default="celeba-128x128", type=str, help="Dataset (possible values: celeba-128x128)")
#parser.add_argument('--datasets_dir', dest="datasets_dir", default="datasets", type=str, help="Location of datasets.")
#parser.add_argument('--color', dest="color", default=True, type=str2bool, help="Color (True/False)")
#parser.add_argument('--train_size', dest="train_size", type=int, default=29000, help="Train set size.")
#parser.add_argument('--test_size', dest="test_size", type=int, default=1000, help="Test set size.")
#parser.add_argument('--shape', dest="shape", default="128,128", help="Image shape.")

# model hyperparameters
args_m = 40
args_alpha = 0.25
args_beta = 0.1
#parser.add_argument('--m', dest="m", type=int, default=120, help="Value of model hyperparameter m.")
#parser.add_argument('--alpha', dest="alpha", type=float, default=0.25, help="Value of model hyperparameter alpha.")
#parser.add_argument('--beta', dest="beta", type=float, default=0.05, help="Value of model hyperparameter beta.")

# training
args_lr = 0.0002
args_batch_size = 64
args_nb_epoch = 200
args_verbose = 1
#parser.add_argument('--lr', dest="lr", default="0.001", type=float, help="Learning rate for the optimizer.")
#parser.add_argument('--batch_size', dest="batch_size", default=200, type=int, help="Batch size.")
#parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=200, help="Number of epochs.")
#parser.add_argument('--verbose', dest="verbose", type=int, default=2, help="Logging verbosity: 0-silent, 1-verbose, 2-perEpoch (default).")

# architecture
args_sampling = True
args_sampling_std = -1.0
args_latent_dim = 256
args_base_filter_num = 16
args_resnet_wideness = 1
#parser.add_argument('--sampling', dest="sampling", type=str2bool, default=True, help="Use sampling.")
#parser.add_argument('--sampling_std', dest="sampling_std", type=float, default=-1.0, help="Sampling std, if < 0, then we learn std.")
#parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension.")
#parser.add_argument('--base_filter_num', dest="base_filter_num", default=32, type=int, help="Initial number of filter in the conv model.")
#parser.add_argument('--resnet_wideness', dest="resnet_wideness", default=1, help="Wideness of resnet model (1-wide first block has 16 filters).")

# encoder
args_encoder_use_bn = True
args_encoder_wd = 0.0
#parser.add_argument('--encoder_use_bn', dest="encoder_use_bn", type=str2bool, default=False, help="Use batch normalization in encoder.")
#parser.add_argument('--encoder_wd', dest="encoder_wd", type=float, default=0.0, help="Weight decay param for the encoder.")

# generator
args_generator_use_bn = True
args_generator_wd = 0.0
#parser.add_argument('--generator_use_bn', dest="generator_use_bn", type=str2bool, default=False, help="Use batch normalization in generator.")
#parser.add_argument('--generator_wd', dest="generator_wd", type=float, default=0.0, help="Weight decay param for generator.")

# locations
args_prefix = 'model/pictures'
args_model_path = 'model/network'
#parser.add_argument('--prefix', dest="prefix", default="trash", help="File prefix for the output visualizations and models.")
#parser.add_argument('--model_path', dest="model_path", default=None, help="Path to saved networks. If None, build networks from scratch.")

# micellaneous
args_memory_share = 0.95
args_frequency = 100
args_save_latent = True
args_latent_cloud_size = 10000
#parser.add_argument('--memory_share', dest="memory_share", type=float, default=0.95, help="Fraction of memory that can be allocated to this process.")
#parser.add_argument('--frequency', dest="frequency", type=int, default=20, help="Image saving frequency.")
#parser.add_argument('--save_latent', dest="save_latent", type=str2bool, default=False, help="If True, then save latent pointcloud.")
#parser.add_argument('--latent_cloud_size', dest="latent_cloud_size", type=int, default=10000, help="Size of latent cloud.")


####################################################################################################
####################################################################################################
####################################################################################################

#args = params.getArgs()
#print(args)

# set random seed
np.random.seed(10)

print('Keras version: ', keras.__version__)
print('Tensorflow version: ', tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args_memory_share
set_session(tf.Session(config=config))

#
# Datasets
#

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

    summary_writer = tf.summary.FileWriter(args_prefix+"/", graph=tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=2)
    #if args_model_path is not None and tf.train.checkpoint_exists(args_model_path):
    #    saver.restore(session, tf.train.latest_checkpoint(args_model_path))
    #    print('Model restored from ' + args_model_path)
    #    ckpt = tf.train.get_checkpoint_state(args_model_path)
    #    global_iters = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
    #    start_epoch = (global_iters * args_batch_size) // args_train_size
    #print('Global iters: ', global_iters)

    for iteration in range(iterations):
        print(iteration)
        epoch = global_iters * args_batch_size // args_train_size
        global_iters += 1

        x = session.run(train_next)
        z_p = np.random.normal(loc=0.0, scale=1.0, size=(args_batch_size, args_latent_dim))
        z_x, x_r, x_p = session.run([z, xr, generator_output], feed_dict={encoder_input: x, generator_input: z_p})

        _ = session.run([encoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
        _ = session.run([generator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
        
        if global_iters % 10 == 0:
            summary, = session.run([summary_op], feed_dict={encoder_input: x})
            summary_writer.add_summary(summary, global_iters)
#aqui!
        if (global_iters % frequency) == 0:
            enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np = \
             session.run([encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, generator_loss, l_ae2, l_reg_zr, l_reg_zpp],
                         feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
            print('Epoch: {}/{}, iteration: {}/{}'.format(epoch+1, args_nb_epoch, iteration+1, iterations))
            print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}'.format(enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np))
            print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zpp: {}'.format(generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np))

        if ((global_iters % iterations_per_epoch == 0) and args_save_latent):
            #utils.save_output(session, args_prefix, epoch, global_iters, args_batch_size, OrderedDict({encoder_input: test_next}), OrderedDict({"test_mean": z_mean, "test_log_var": z_log_var}), args_test_size)
            #utils.save_output(session, args_prefix, epoch, global_iters, args_batch_size, OrderedDict({encoder_input: fixed_next}), OrderedDict({"train_mean": z_mean, "train_log_var": z_log_var}), args_latent_cloud_size)

            n_x = 5
            n_y = args_batch_size // n_x
            print('Save original images.')
            utils.plot_images(np.transpose(x, (0, 2, 3, 1)), n_x, n_y, "{}_original_epoch{}_iter{}".format(args_prefix, epoch + 1, global_iters), text=None)
            print('Save generated images.')
            utils.plot_images(np.transpose(x_p, (0, 2, 3, 1)), n_x, n_y, "{}_sampled_epoch{}_iter{}".format(args_prefix, epoch + 1, global_iters), text=None)
            print('Save reconstructed images.')
            utils.plot_images(np.transpose(x_r, (0, 2, 3, 1)), n_x, n_y, "{}_reconstructed_epoch{}_iter{}".format(args_prefix, epoch + 1, global_iters), text=None)

        if ((global_iters % iterations_per_epoch == 0) and ((epoch + 1) % 10 == 0)):
            if args_model_path is not None:
                saved = saver.save(session, args_model_path + "/model", global_step=global_iters)
                print('Saved model to ' + saved)
