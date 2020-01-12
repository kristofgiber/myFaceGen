
# Author: Kristof Giber (2019 September)
# This is my solution for disentangled VAE image generation in TF2 with functional API
# Works with both celebA and MNIST data.
# Includes options for distributed training (tf.distribute.MirroredStrategy) on physical GPUs (and allows simulation of multi-GPU behavior on a single CPU for test purpose)
# Allows both the training and evaluation, such as the generation of new faces from random prior sample or altering specific features (eg. smile) in a fixed test input sample ('reconstruction')
# Various visualization options of how the generated faces or digits evolve during training (plot clustering for 2D space or manifold traversal for 2+ dimenstions)

# To track training with tensorboard through remote server,
# first launch tensorboard on the server via a port-forwarded ssh session, then open the port in local browser:
# ssh -L 16006:127.0.0.1:6006 user@ip
# tensorboard --logdir=/home/user/VAE/MNIST/logs
# in browser go to: localhost:16006/

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers.distribution_layer
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers


########  MAIN PARAMETERS  #########

dataset = 'celeba'  # 'mnist' | 'celeba'
use_small_dataset_forCelebA = False

latent_dim = 16  # 8  # 50
ndims_to_plot = 16  # latent_dim

BATCH_SIZE_per_replica = 200  # 100  # 12
TEST_BATCH_SIZE_per_replica = 100  # 250
learning_rate = 1e-4  # 1e-3
dense_hidden_units = 500  # 500
beta = 4.0
reconstruction_weight = 1.0

# use_discriminator = True
use_discriminator = False
if use_discriminator:
    beta = 1.0
    reconstruction_weight = 0.5

## settings for mnist:
if dataset == 'mnist':
    network_type = 'dense'  # 'conv' | 'dense'
    reconstruction_img_rows = 0
    reconstruction_img_cols = 0
elif dataset == 'celeba':
    network_type = 'conv'  # 'conv' | 'dense'
    reconstruction_img_rows = 0 # 10  # 4
    reconstruction_img_cols = 0 # 5  # 2

save_each_validation_as_separate_image = 1  # 0: overwrite image saved in previous step.  1: keeps all images (allows video making)
big_validation_frequency = 5   # Note This is all the slow stuff that we want rarely, image reconstructions / generation, histogram logging... Set it rare (eg. 10 or 100 meaning at every 10-100 epoch.). In addition to this, a quick validation defined by validation_split and ran by keras already runs every epoch
validation_method = 'reconstruct'  # 'reconstruct' | 'generate'

# Note tfp option currently doesn't work due to a bug with tfp 0.8-rc0 the encoder prediction (ie. .predict() with a model that outputs a tfp layer defaulting to convert_to_tensor_fn sample())
# see my reported issue: https://github.com/tensorflow/tensorflow/issues/32219#issuecomment-528224103
use_tensorflow_probability = False

caching = 'basic'  # 'basic' | 'tfrecord' | '' empty string means no caching nor tfrecords
cache_path = '/mnt/'  # './'  # note: for celeba 200K images to cache with batches of 100, it takes about 74Gb space so make sure to use a large attached disk not the boot disk as path (eg. on Azure: '/mnt/')

load_model = 0
model_load_mode = 'full_model'  # 'full_model' | 'weights_only'
model_load_path = 'model'
model_load_name = 'Model'

# Note: 'full_model' saves the entire model in .h5 format (this is what we need to load it from tf.js, on a browser or mobile)
save_model = 1
model_save_weights_only = False  # Directly callback input to keras model.fit() save_weights_only flag
model_save_path = 'Models/vae_model'

simulated_gpus_for_distributed_testing = 0
if simulated_gpus_for_distributed_testing:
    from VAE import performance_tools as perf
    perf.simulate_multiGPUs(simulated_gpus_for_distributed_testing, memorylimit=2048)

# This will fix seed for all random sampling including, np, tf and tfd (for tfd we add the seed=fixed_seed to each .sample() call)
# Note: this is only overridden locally in the get_one_of_each_mnist_digit() method to find a seed that yields a representative set of digits
fixed_seed = 1
if fixed_seed:
    np.random.seed(fixed_seed)
    tf.random.set_seed(fixed_seed)

debugging = 0
if debugging:
    tf.config.experimental_run_functions_eagerly(True)

# has to be set at program startup:
strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync
print ('Number of replicas: {}'.format(num_replicas))
BATCH_SIZE = BATCH_SIZE_per_replica * num_replicas
TEST_BATCH_SIZE = TEST_BATCH_SIZE_per_replica * num_replicas
print("Number of replicas: {}\n"
      "global training batch size: {}; training batch size per replica: {}\n"
      "global test batch size: {}; test batch size per replica: {}\n"
      .format(num_replicas, BATCH_SIZE, BATCH_SIZE_per_replica, TEST_BATCH_SIZE, TEST_BATCH_SIZE_per_replica))

plt.style.use('default')


### LOADING DATASET ###

if dataset == 'mnist':
    outdir = 'MNIST/'
    color_channels = 1
    (train_images, _), (test_images, test_labels) = tfk.datasets.mnist.load_data()
    print("Loaded MNIST dataset.")
    # we have 60,000 training images
    TRAIN_BUF = n_trainsamples = np.shape(train_images)[0]
    print('Number of training samples: {}'.format(n_trainsamples))
    # 10,000 test images
    TEST_BUF = n_testsamples = np.shape(test_images)[0]
    print('Number of test samples: {}'.format(n_testsamples))
    # and the image width and height are both 28 pixels
    imsize = np.shape(train_images)[1]
    print('Width and height of samples: {}'.format(imsize))
    # preproc
    np.random.shuffle(train_images)
    train_images = train_images.reshape(-1, imsize, imsize, 1).astype('float32')
    test_images = test_images.reshape(-1, imsize, imsize, 1).astype('float32')
    train_images /= 255.
    test_images /= 255.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_images)).shuffle(TRAIN_BUF).repeat().batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_images)).shuffle(TEST_BUF).repeat().batch(TEST_BATCH_SIZE)
    def get_one_of_each_mnist_digit(fixed_seed=fixed_seed):
        # we can define any seed value to make sure the 10 digits to be reconstructed are all representative to their class
        np.random.seed(fixed_seed)
        nx = 10
        samples = np.zeros([nx, imsize, imsize, color_channels]).astype('float32')
        diglabels = range(nx)
        for i, xi in enumerate(diglabels):
            digits = [k for k, d in enumerate(test_labels) if d == xi]
            digit_ind = np.random.choice(digits)
            print("digit {} digit index: {}".format(xi, digit_ind))
            samples[i, :, :, 0] = test_images[digit_ind, :, :, 0]
        return samples
    # get a sample of each digits for reconstruction (we want these to be the same across training to track progress)
    reconstructables = get_one_of_each_mnist_digit(fixed_seed=3)



elif dataset == 'celeba':
    outdir = 'CelebA/'
    color_channels = 3
    datadir = 'CelebA/data/img_align_celeba/'
    if use_small_dataset_forCelebA:
        datadir_train = datadir + 'train_small/'
        datadir_test = datadir + 'test_small/'
    else:
        datadir_train = datadir + 'train/'
        datadir_test = datadir + 'test/'

    imsize = 128

    TRAIN_BUF = 10000  # ideally should be the number of images used per epoch ie. all images ie. 200000
    TEST_BUF = 10000
    img_shape = (imsize, imsize, 3)
    # Note the size of each image in celeba: 178x218.
    # There are 202599 images. I have placed the last 2599 images in the 'test' folder and 200K to the 'train' folder

    def input_pipeline(batch_size, buffer_size=1000, dataname='', image_dir='',  label_dir=None):
        image_files = [image_dir + f for f in np.sort(os.listdir(image_dir))]  # relative path
        num_samples = len(image_files)
        print("\n{} images found in {} data directory".format(num_samples, dataname))
        def load_and_preprocess_image(path):
            # TODO: see if there is a way to pass batches in here for loading instead of single paths
            # Doc says there's a way to 'vectorize' pass batches into the map function.
            # Maybe I'd just need to call .batch() before .map(). But that may have other downsides.
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [imsize, imsize])
            image /= 255.0  # normalize to [0,1] range
            return image
        path_ds = tf.data.Dataset.from_tensor_slices(image_files)
        image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = tf.data.Dataset.zip((image_ds, image_ds))

        if caching=='basic':
            ds = ds.cache(filename=cache_path + 'cache.tf-data')  # or just image_ds.cache() to cache in memory
        elif caching == 'tfrecord':
            ds = ds.map(tf.serialize_tensor)
            tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
            tfrec.write(ds)
            ds = tf.data.TFRecordDataset('images.tfrec')
            def parse_tfrec(x):
                result = tf.parse_tensor(x, out_type=tf.float32)
                result = tf.reshape(result, img_shape)
                return result
            ds.map(parse_tfrec, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if fixed_seed:
            ds = ds.shuffle(buffer_size=buffer_size, seed=fixed_seed).repeat()
        else:
            ds = ds.shuffle(buffer_size=buffer_size).repeat()
        ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds, image_files
    train_dataset, train_image_files = input_pipeline(BATCH_SIZE, TRAIN_BUF, 'train', datadir_train)
    test_dataset, test_image_files = input_pipeline(TEST_BATCH_SIZE, TEST_BUF, 'test', datadir_test)
    n_trainsamples = len(train_image_files)
    n_testsamples = len(test_image_files)

    def get_reconstructable_faces(fixed_seed=fixed_seed):
        # we can define any seed value to make sure the 10 digits to be reconstructed are all representative to their class
        np.random.seed(fixed_seed)
        nx = 8  # 4
        samples = np.zeros([nx, imsize, imsize, color_channels]).astype('float32')
        for i in range(nx):
            file_to_load = np.random.choice(test_image_files)
            print("face image name: {}".format(file_to_load))
            samples[i, ::] = tfk.preprocessing.image.load_img(file_to_load, target_size=img_shape)
        samples /= 255.0
        return samples
    reconstructables = get_reconstructable_faces(fixed_seed=3)

if not os.path.exists(outdir):
    os.makedirs(outdir)
logdir = outdir + "logs"  # "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(logdir)
model_save_path = outdir + model_save_path
model_load_path = outdir + model_load_path

if 0 < reconstruction_img_rows * reconstruction_img_cols < ndims_to_plot:
    print("Warning: Ignoring user input 'reconstruction_img_rows' and 'reconstruction_img_cols'"
          "because {} * {} = {} but we need at least ndims_to_plot = {} cells to plot all images"
          .format(reconstruction_img_rows, reconstruction_img_cols, reconstruction_img_rows * reconstruction_img_cols, ndims_to_plot))
    reconstruction_img_rows = reconstruction_img_cols = 0




#####################################################################################
################   PLOTTING AND EVALUATION FUNCTIONS   ##############################
#####################################################################################

def plot_2D_clusters(epoch, encoder_model):
    # plot clustering:
    plt.figure(figsize=(8, 6))
    dpi = 300
    plt.title("Epoch {:04d}".format(epoch))
    codes = encoder_model.predict(test_images)
    plt.scatter(codes[:, 0], codes[:, 1], c=test_labels, edgecolors='k')  # alpha=0.8)
    plt.set_cmap('jet')
    plt.colorbar()
    plt.grid()
    outputfolder = outdir + 'Figs_Generated2dim/'
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    plt.savefig(outputfolder + "VAE_clusters_2D.png", dpi=dpi)
    plt.close()

def plot_2D_traversal(epoch, decoder_model):
    # plot manifold traversal:
    plt.figure(figsize=(8, 6))
    dpi = 300
    plt.title("Epoch {:04d}".format(epoch))
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.zeros((imsize * ny, imsize * nx, color_channels))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
          z_mu = np.array([[xi, yi]])
          x_mean = decoder_model.predict(z_mu)
          canvas[(nx - i - 1) * imsize:(nx - i) * imsize, j * imsize:(j + 1) * imsize] = x_mean
    np.meshgrid(x_values, y_values)
    canvas = canvas.squeeze()
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    outputfolder = outdir + 'Figs_Generated2dim/'
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    plt.savefig(outputfolder + "VAE_B{:.1f}_latent_traversal_2dim_generated.png".format(beta), dpi=dpi)
    plt.close()


def plot_multidim_traversal_generate(epoch, decoder_model):
    # # traversing through selected latent dimensions while fixing the rest at 0:
    # plot manifold traversal:
    plt.figure(figsize=(8, 6))
    dpi = 300
    plt.title("Epoch {:04d}".format(epoch))
    nx = 20
    dims = range(latent_dim)
    y_values = np.linspace(-3, 3, nx)
    canvas = np.zeros((imsize * latent_dim, imsize * nx, color_channels)).squeeze()
    for i, xi in enumerate(dims):
        for j, yi in enumerate(y_values):
            latent_vectors_np = [0.]*latent_dim
            latent_vectors_np[xi] = yi
            z_mu = np.array([latent_vectors_np])
            x_mean = np.squeeze(decoder_model.predict(z_mu))
            canvas[j + imsize*i:j + imsize*i + imsize, j + i * imsize: j + i * imsize + imsize] = x_mean
    np.meshgrid(dims, y_values)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    outputfolder = outdir + 'Figs_GeneratedMultidim/'
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    plt.savefig(outputfolder + "VAE_B{:.1f}_LatentTraversal_multidim_generated.png".format(beta), dpi=dpi)
    plt.close()


# def plot_multidim_traversal_reconstruct(samples, reconstructor_model, elbo, epoch, output_file_index=0):
def plot_multidim_traversal_reconstruct(samples, encoder_model, decoder_model, elbo, epoch, output_file_index=0):
    # Each subplot shows traversal through a single dimension while leaving the other dimensions unmodified:
    # Argument 'samples' should be in (sample, imsize, imsize) format.
    # Argument 'output_file_index' (only used when save_each_validation_as_separate_image==True)
    dims = np.linspace(0, latent_dim-1, ndims_to_plot, dtype='int32')
    # dims = range(latent_dim)
    if reconstruction_img_rows:
        nrows = reconstruction_img_rows
    else:
        nrows = int(np.sqrt(ndims_to_plot))
    if reconstruction_img_cols:
        ncols = reconstruction_img_cols
    else:
        ncols = int(np.ceil(ndims_to_plot/int(np.sqrt(ndims_to_plot))))
    plt.figure(figsize=(8, 6))
    dpi = 300
    if color_channels > 1:
        dpi += 100
    if ndims_to_plot > 12:
        dpi += 100
    if ndims_to_plot > 40:
        dpi += 100
    plt.title("Epoch {:04d} beta: {:.1f} lr: {:.0e} ELBO: {:.3f}".format(epoch, beta, learning_rate, elbo))
    nsamples = samples.shape[0]
    ntrav = 8  # 10
    traversal_values = np.linspace(-3, 3, ntrav)
    pad = 4
    left_side_ribbon = pad + imsize + pad  # space for the original input images displayed before each row
    canvas = np.ones((imsize * nsamples * nrows + pad * (nrows+1), left_side_ribbon + imsize * ntrav * ncols + pad * (ncols+1), color_channels))
    for row in range(nrows):
        for j in range(nsamples):
            inputimage = samples[j, ::]
            y_startpos = pad + row * (imsize * nsamples + pad) + j * imsize
            x_startpos = pad
            canvas[y_startpos:y_startpos + imsize, x_startpos:x_startpos + imsize, :] = inputimage
    for i, dim in enumerate(dims):
        for k, zmod in enumerate(traversal_values):
            latent_codes = encoder_model.predict(samples)
            latent_codes[:, dim] += zmod
            reconstructed_ims = decoder_model.predict(latent_codes)
            for j in range(nsamples):
                row = i // ncols
                col = i % ncols
                y_startpos = pad + row * (imsize * nsamples + pad) + j * imsize
                x_startpos = left_side_ribbon + pad + col * (imsize * ntrav + pad) + k * imsize
                canvas[y_startpos:y_startpos+imsize, x_startpos:x_startpos+imsize, :] = reconstructed_ims[j, ::]
    plt.imshow(np.squeeze(canvas), origin="upper")
    if color_channels == 1:
        plt.set_cmap('gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    addtoname = ''
    if save_each_validation_as_separate_image:
        addtoname = '_frame{:04d}'.format(output_file_index)
    outputfolder = outdir + 'Figs_Reconstructed/'
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    plt.savefig(outputfolder + "VAE_B{:.1f}_LatentTraversal_multidim_reconstruct{}.png".format(beta, addtoname), dpi=dpi)
    plt.close()




##############################################################################################################
########################  Distribute strategy context for architecture, optimizer and training loop  ##########
##############################################################################################################

with strategy.scope():

    ###  DEFINING MODEL architecture and loss using Keras Functional API  ###
    ################  TRAINING AND EVALUATION  ##############################

    # For a list of standard callbacks:
    # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/callbacks

    ######### Custom callbacks for training evaluation:  ###############

    class CustomValidations(tfk.callbacks.Callback):
        def __init__(self, latent_dimensions=0, # colchans=0, image_size=0,
                     images_to_reconstruct=None,  # reconstructor_model=None,
                     encoder_model=None, decoder_model=None):
            super().__init__()
            self.valcount = 0
            self.start_time_total = time.time()
            self.images_to_reconstruct = images_to_reconstruct
            self.encoder_model = encoder_model
            self.decoder_model = decoder_model
            self.latent_dim = latent_dimensions

        def on_epoch_end(self, epoch, logs=None):
            if epoch>0 and (epoch+1) % big_validation_frequency == 0:
                start_time_image = time.time()
                print('\tThe average loss for epoch {} is {:.4f}.'.format(epoch+1, logs['val_loss'].mean()))
                if validation_method == 'reconstruct':
                    print('Reconstructing test images...')
                    plot_multidim_traversal_reconstruct(self.images_to_reconstruct, self.encoder_model, self.decoder_model, logs['val_loss'].mean(), epoch+1, self.valcount)
                elif validation_method == 'generate':
                    print('Generating test images...')
                    if self.latent_dim == 2:
                        plot_2D_clusters(epoch+1, self.encoder_model)
                        plot_2D_traversal(epoch+1, self.decoder_model)
                    else:
                        plot_multidim_traversal_generate(epoch+1, self.decoder_model)
                self.valcount += 1
                time_image = time.time() - start_time_image
                print("Images decoded and saved ({:.2f}sec).".format(time_image))
                total_training_time = time.time() - self.start_time_total
                print("Total training time: {:.2f}min ({:.2f}h)".format(total_training_time / 60.,
                                                                        total_training_time / 3600.))

    optimizer = tfk.optimizers.Adam(learning_rate)
    # the input layer is shared between all models and architecture types
    image_input = tfk.Input(shape=(imsize, imsize, color_channels), name='encoder_input')

    if load_model and model_load_mode == 'full_model':
        model = tfk.experimental.load_from_saved_model(model_load_path + '/' + model_load_name)
        # The model will be compiled normally.

    else:

        ###############  MODEL ARCHITECTURE  ###################

        class Sampling(tfkl.Layer):
            """Uses (z_mean, z_log_var) to sample z, the vector encoding an image."""
            def call(self, inputs):
                z_mean, z_log_var = inputs
                epsilon = tfk.backend.random_normal(shape=tf.shape(z_mean))  # normal function defaults to mean 0 and std 1 thus, we ensure standard gaussian distribution for the estimated posterior q(z | x)
                return z_mean + tfk.backend.exp(.5 * z_log_var) * epsilon  # equals eps * sqrt(var) + mean (ie., eps * std + mean) because: np.exp(np.log(x) * .5) = x ** .5 = np.sqrt(x)

        # Define encoder model (inference network)

        if network_type == 'dense':
            x = tfkl.Flatten()(image_input)
            x = tfkl.Dense(dense_hidden_units, activation='softplus', name="Inference-l1_Dense")(x)
            x = tfkl.Dense(dense_hidden_units, activation='softplus', name="Inference-l2_Dense")(x)
        else:
            if dataset == 'mnist':
                x = tfkl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='softplus',
                                           name='Inference-l1_Conv2D')(image_input)
                x = tfkl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='softplus',
                                       name="Inference-l2_Conv2D")(x)
                x = tfkl.Flatten()(x)
            elif dataset == 'celeba':
                x = tfkl.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='softplus',
                                       name="Inference-l1_Conv2D")(image_input)
                # tfkl.BatchNormalization(),
                x = tfkl.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='softplus',
                                       name="Inference-l2_Conv2D")(x)
                # tfkl.BatchNormalization(),
                # tfkl.MaxPool2D(pool_size=(2, 2)),
                x = tfkl.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='softplus',
                                       name="Inference-l3_Conv2D")(x)
                # tfkl.BatchNormalization(),
                x = tfkl.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='softplus',
                                       name="Inference-l4_Conv2D")(x)
                # tfkl.BatchNormalization(),
                x = tfkl.Flatten()(x)

        if use_tensorflow_probability:
            x = tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim))(x)
            z = tfpl.MultivariateNormalTriL(latent_dim)(x)
        else:
            z_mean = tfkl.Dense(latent_dim, name='z_mean')(x)
            z_log_var = tfkl.Dense(latent_dim, name='z_log_var')(x)
            z = Sampling(name="z")((z_mean, z_log_var))

        encoder = tfk.Model(inputs=image_input, outputs=z, name='encoder')
        if use_discriminator:
            encoder_with_distribution = tfk.Model(inputs=image_input, outputs=[z_mean, z_log_var, z], name='encoder_with_distribution')


        # Define decoder model (generative network)
        latent_inputs = tfk.Input(shape=(latent_dim,), name='z_sampling')

        if network_type == 'dense':
            x = tfkl.Dense(dense_hidden_units, activation='softplus', name="Generative-l1_Dense")(latent_inputs)
            x = tfkl.Dense(dense_hidden_units, activation='softplus', name="Generative-l2_Dense")(x)
            x = tfkl.Dense(imsize ** 2 * color_channels, activation='sigmoid', name="Generative-l3_Dense_out")(x)  # , activation='sigmoid')(x)
            output_probs = tfkl.Reshape(target_shape=(imsize, imsize, color_channels), name="Generative-output_probs")(x)
        else:
            if dataset == 'mnist':
                x = tfkl.Dense(units=7 * 7 * 32 * color_channels, activation=tf.nn.softplus,
                                      name="Generative-l1_DenseRelu")(latent_inputs)
                x = tfkl.Reshape(target_shape=(7, 7, 32 * color_channels))(x)
                x = tfkl.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                                activation='softplus', name="Generative-l2_Conv2DTranspose")(x)
                x = tfkl.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                                activation='softplus', name="Generative-l3_Conv2DTranspose")(x)
                # No activation
                output_probs = tfkl.Conv2DTranspose(filters=color_channels, kernel_size=(3, 3), strides=(1, 1),
                                                padding="SAME", activation='sigmoid', name="Generative-l4_Conv2DTranspose-output_probs")(x)
            elif dataset == 'celeba':
                x = tfkl.Dense(units=8 * 8 * 64 * color_channels, activation=tf.nn.softplus,
                                      name="Generative-l1_DenseRelu")(latent_inputs)
                x = tfkl.Reshape(target_shape=(8, 8, 64 * color_channels))(x)
                x = tfkl.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                                activation='softplus', name="Generative-l2_Conv2DTranspose")(x)
                # tfkl.BatchNormalization(),
                x = tfkl.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                                activation='softplus', name="Generative-l3_Conv2DTranspose")(x)
                # tfkl.BatchNormalization(),
                x = tfkl.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                                activation='softplus', name="Generative-l4_Conv2DTranspose")(x)
                # tfkl.BatchNormalization(),
                x = tfkl.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                                activation='softplus', name="Generative-l5_Conv2DTranspose")(x)
                # No activation
                # tfkl.BatchNormalization(),
                output_probs = tfkl.Conv2DTranspose(filters=color_channels, kernel_size=(3, 3), strides=(1, 1),
                                                padding="SAME", activation='sigmoid', name="Generative-l6_Conv2DTranspose-output_probs")(x)


        decoder = tfk.Model(inputs=latent_inputs, outputs=output_probs, name='decoder')

        # Define VAE model.
        output_probs = decoder(z)

        vae_model = tfk.Model(inputs=image_input, outputs=output_probs, name='vae')

        if use_tensorflow_probability:
            prior = tfd.Independent(tfd.Normal(loc=[0., 0], scale=1), reinterpreted_batch_ndims=1)
            tfpl.KLDivergenceAddLoss(prior, weight=beta)
        else:
            def loss_fn(ypred, ytrue):
                cross_entropy = tfk.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                reconstruction_loss = reconstruction_weight * tf.reduce_sum(cross_entropy(ypred, ytrue)) * (1. / BATCH_SIZE)
                return reconstruction_loss

            log2pi = tfk.backend.log(2. * np.pi)
            logpz = tfk.backend.sum(-.5 * (tf.square(z) + log2pi), axis=1)  # the prior of z has normal distribution with zero mean and 0 logvar (var=1)
            logqz_x = tfk.backend.sum(-.5 * (tf.square(z - z_mean) * tf.exp(-z_log_var) + z_log_var + log2pi), axis=1)  # the learned posterior of z has normal distribution and mean and logvar is trained to approximate the above prior
            kl_loss = - beta * (logpz - logqz_x)
            if not use_discriminator:
                vae_model.add_loss(kl_loss)

        if use_discriminator:
            # compute KL loss for both term
            alpha = 0.25
            margin = 110.  # 90.

            # Encoder loss:
            x_stop_grad = tfkl.Lambda(lambda x: tfk.backend.stop_gradient(x))(output_probs)
            z_mean, z_log_var, z_rec = encoder_with_distribution(x_stop_grad)
            logpz = tfk.backend.sum(-.5 * (tf.square(z_rec) + log2pi), axis=1)
            logqz_x = tfk.backend.sum(-.5 * (tf.square(z_rec - z_mean) * tf.exp(-z_log_var) + z_log_var + log2pi), axis=1)
            kl_loss_Zr = - (logpz - logqz_x)

            prior_sampling = tfk.backend.random_normal(shape=tf.shape(z_mean))
            decoded = decoder(prior_sampling)
            z_mean, z_log_var, z_rec = encoder_with_distribution(decoded)
            logpz = tfk.backend.sum(-.5 * (tf.square(z_rec) + log2pi), axis=1)
            logqz_x = tfk.backend.sum(-.5 * (tf.square(z_rec - z_mean) * tf.exp(-z_log_var) + z_log_var + log2pi), axis=1)
            kl_loss_Zp = - (logpz - logqz_x)

            encoder_loss = alpha * (tf.math.maximum(margin-kl_loss_Zr, 0.) + tf.math.maximum(margin-kl_loss_Zp, 0.))
            generative_loss = alpha * (tf.math.maximum(kl_loss_Zr, 0.) + tf.math.maximum(kl_loss_Zp, 0.))

            encoder.add_loss(kl_loss)
            encoder.add_loss(encoder_loss)
            encoder.add_loss(loss_fn(output_probs, image_input))

            vae_model.add_loss(generative_loss)


    vae_model.compile(optimizer, loss_fn)

    vae_model.metrics.append(kl_loss)
    vae_model.metrics_names.append("kl_loss")


    ########  Standard built-in callbacks for training evaluation, model saving and logging:  ########

    callbacks_list =[]
    if save_model:
        checkpoint = tfk.callbacks.ModelCheckpoint(
            filepath=model_save_path,  # +".{epoch:04d}",  #-{val_loss:.3f}",
            monitor='val_loss',  # 'loss'
            verbose=1,
            mode='min',
            save_weights_only=model_save_weights_only,
            save_freq='epoch'
        )
        callbacks_list.append(checkpoint)
    callbacks_list.append(CustomValidations(latent_dimensions=latent_dim,
                                            images_to_reconstruct=reconstructables,
                                            encoder_model=encoder, decoder_model=decoder))
    # There is also remote monitor standard callback to send certain training updates to target remote machine, using reqests module: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/callbacks/RemoteMonitor

    history = vae_model.fit(train_dataset, steps_per_epoch=n_trainsamples//BATCH_SIZE, validation_data=test_dataset, validation_steps=n_testsamples//TEST_BATCH_SIZE, epochs=9999999, callbacks=callbacks_list)







