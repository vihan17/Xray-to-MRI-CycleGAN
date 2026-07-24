import random
import time
import tensorflow as tf
from matplotlib import pyplot as plt

default_initializer = tf.random_normal_initializer(0., 0.02)
generator_default_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_default_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


class LayerConfiguration:
    def __init__(self, filters, kernel_size=4):
        self.filters = filters
        self.kernel_size = kernel_size


class DownsampleLayerConfiguration(LayerConfiguration):
    def __init__(self, filters, kernel_size=4, apply_norm=True):
        super().__init__(filters, kernel_size)
        self.apply_norm = apply_norm


class UNetLayerConfiguration(DownsampleLayerConfiguration):
    def __init__(self, filters, kernel_size=4, apply_norm=True, dropout=0.0):
        super().__init__(filters, kernel_size, apply_norm)
        self.dropout = dropout


u_net_encoder_decoder_layers = tuple([UNetLayerConfiguration(64, 4, apply_norm=False),
                                      UNetLayerConfiguration(128, 4),
                                      UNetLayerConfiguration(256, 4),
                                      UNetLayerConfiguration(512, 4),
                                      UNetLayerConfiguration(512, 4, dropout=0.5),
                                      UNetLayerConfiguration(512, 4, dropout=0.5),
                                      UNetLayerConfiguration(512, 4, dropout=0.5)])
u_net_connecting_layers = tuple([DownsampleLayerConfiguration(512, 4)])

discriminator_downsampling_layers = tuple([DownsampleLayerConfiguration(64, 4, apply_norm=False),
                                           DownsampleLayerConfiguration(128, 4),
                                           DownsampleLayerConfiguration(256, 4)])
discriminator_final_convolution_layer = LayerConfiguration(512, 4)


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
        self.scale = None
        self.offset = None

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(1., 0.02)
        self.scale = self.add_weight(name='scale', shape=input_shape[-1:], initializer=initializer, trainable=True)
        self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def downsample(filters, kernel_size, apply_norm):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same',
                                      kernel_initializer=default_initializer, use_bias=False))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, kernel_size, dropout):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                                               kernel_initializer=default_initializer, use_bias=False))
    result.add(InstanceNormalization())
    if 0 < dropout < 1:
        result.add(tf.keras.layers.Dropout(dropout))
    result.add(tf.keras.layers.ReLU())
    return result


def u_net_generator(output_channels=3, input_shape=(256, 256, 3), encoder_decoder_layers=u_net_encoder_decoder_layers,
                    connecting_layers=u_net_connecting_layers, kernel_size_last_layer=4):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    skips = []
    for layer_configuration in encoder_decoder_layers:
        down = downsample(layer_configuration.filters, layer_configuration.kernel_size, layer_configuration.apply_norm)
        x = down(x)
        skips.append(x)

    for layer_configuration in connecting_layers:
        down = downsample(layer_configuration.filters, layer_configuration.kernel_size, layer_configuration.apply_norm)
        x = down(x)

    for i in reversed(range(len(encoder_decoder_layers))):
        layer_configuration = encoder_decoder_layers[i]
        up = upsample(layer_configuration.filters, layer_configuration.kernel_size, layer_configuration.dropout)
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skips[i]])

    last = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size_last_layer, strides=2, padding='same',
                                           kernel_initializer=default_initializer, activation='tanh')
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(input_shape=(256, 256, 3), downsampling_layers=discriminator_downsampling_layers,
                  final_convolution_layer=discriminator_final_convolution_layer, kernel_size_last_layer=4):
    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')

    x = inp
    for layer_configuration in downsampling_layers:
        down = downsample(layer_configuration.filters, layer_configuration.kernel_size, layer_configuration.apply_norm)
        x = down(x)

    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(final_convolution_layer.filters, final_convolution_layer.kernel_size, strides=1,
                               kernel_initializer=default_initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)

    last = tf.keras.layers.Conv2D(1, kernel_size_last_layer, strides=1, kernel_initializer=default_initializer)(x)

    return tf.keras.Model(inputs=inp, outputs=last)


def plot_transformations(image_sample_x, image_sample_y, generator_x2y, generator_y2x, distributions_names=('X', 'Y'),
                         fig_size=(12, 12), contrast=1, main_title="Mapping between X->Y and Y->X", save_as=None):
    plt.figure(figsize=fig_size)
    plt.suptitle(main_title, fontsize="x-large")

    def plot_row(image_sample, generator, generator_other, row_index):
        base_index = (row_index + 1) * (row_index + 1)

        plt.subplot(2, 3, base_index)
        plt.title("Original sample from '{}'".format(distributions_names[row_index % 2]))
        plt.imshow(image_sample * 0.5 * contrast + 0.5)

        plt.subplot(2, 3, base_index + 1)
        plt.title("Converted sample from '{}' to '{}'"
                  .format(distributions_names[row_index % 2], distributions_names[(row_index + 1) % 2]))
        generated_img = generator(tf.cast([image_sample], tf.float32))[0]
        plt.imshow(generated_img * 0.5 * contrast + 0.5)

        plt.subplot(2, 3, base_index + 2)
        plt.title("Converted sample from '{}' to '{}'"
                  .format(distributions_names[row_index % 2], distributions_names[row_index % 2]))
        generated_img = generator_other(tf.cast([image_sample], tf.float32))[0]
        plt.imshow(generated_img * 0.5 * contrast + 0.5)

    plot_row(image_sample_x, generator_x2y, generator_y2x, 0)
    plot_row(image_sample_y, generator_y2x, generator_x2y, 1)

    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


def get_instance_data_set_generator(data_set, function_to_apply_per_instance=None):
    while True:
        shuffled_indexes = list(range(len(data_set)))
        random.shuffle(shuffled_indexes)

        for index in shuffled_indexes:
            instance = data_set[index]
            if function_to_apply_per_instance is not None:
                instance = function_to_apply_per_instance(instance)
            yield instance


def create_training_function(data_set_generator_x, data_set_generator_y, number_of_samples, generator_x2y,
                             discriminator_x, generator_y2x, discriminator_y,
                             generator_x2y_optimizer=generator_default_optimizer,
                             generator_y2x_optimizer=generator_default_optimizer,
                             discriminator_x_optimizer=discriminator_default_optimizer,
                             discriminator_y_optimizer=discriminator_default_optimizer,
                             lambda_p=10, use_identity_loss=True, summary_writer=None):
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real, generated):
        real_loss = loss_obj(tf.ones_like(real), real)
        generated_loss = loss_obj(tf.zeros_like(generated), generated)
        return (real_loss + generated_loss) * 0.5

    def generator_loss(generated):
        return loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(real_image, cycled_image):
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return lambda_p * loss

    def identity_loss(real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return lambda_p * 0.5 * loss

    @tf.function
    def train_step(real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            fake_y = generator_x2y(real_x, training=True)
            cycled_x = generator_y2x(fake_y, training=True)
            disc_real_x_validity = discriminator_x(real_x, training=True)
            disc_fake_y_validity = discriminator_y(fake_y, training=True)

            fake_x = generator_y2x(real_y, training=True)
            cycled_y = generator_x2y(fake_x, training=True)
            disc_real_y_validity = discriminator_y(real_y, training=True)
            disc_fake_x_validity = discriminator_x(fake_x, training=True)

            total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
            total_gen_x2y_loss = generator_loss(disc_fake_y_validity) + total_cycle_loss
            total_gen_y2x_loss = generator_loss(disc_fake_x_validity) + total_cycle_loss
            if use_identity_loss:
                same_x = generator_y2x(real_x, training=True) if use_identity_loss else None
                same_y = generator_x2y(real_y, training=True)

                total_gen_x2y_loss = total_gen_x2y_loss + identity_loss(real_y, same_y)
                total_gen_y2x_loss = total_gen_y2x_loss + identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(disc_real_x_validity, disc_fake_x_validity)
            disc_y_loss = discriminator_loss(disc_real_y_validity, disc_fake_y_validity)

        generator_x2y_gradients = tape.gradient(total_gen_x2y_loss, generator_x2y.trainable_variables)
        generator_x2y_optimizer.apply_gradients(zip(generator_x2y_gradients, generator_x2y.trainable_variables))

        generator_y2x_gradients = tape.gradient(total_gen_y2x_loss, generator_y2x.trainable_variables)
        generator_y2x_optimizer.apply_gradients(zip(generator_y2x_gradients, generator_y2x.trainable_variables))

        discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))

        discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

        return total_gen_x2y_loss, disc_x_loss, total_gen_y2x_loss, disc_y_loss

    def run_epoch(epoch):
        start = time.time()
        counter = 0

        avg_total_gen_x2y_loss = tf.keras.metrics.Mean(name='total_gen_x2y_loss', dtype=tf.float32)
        avg_total_gen_y2x_loss = tf.keras.metrics.Mean(name='total_gen_y2x_loss', dtype=tf.float32)
        avg_disc_x_loss = tf.keras.metrics.Mean(name='disc_x_loss', dtype=tf.float32)
        avg_disc_y_loss = tf.keras.metrics.Mean(name='disc_y_loss', dtype=tf.float32)

        while counter < number_of_samples:
            counter += 1

            real_x = next(data_set_generator_x)
            real_y = next(data_set_generator_y)

            total_gen_x2y_loss, disc_x_loss, total_gen_y2x_loss, disc_y_loss = train_step(real_x, real_y)

            avg_total_gen_x2y_loss.update_state(total_gen_x2y_loss)
            avg_total_gen_y2x_loss.update_state(total_gen_y2x_loss)
            avg_disc_x_loss.update_state(disc_x_loss)
            avg_disc_y_loss.update_state(disc_y_loss)

        if summary_writer is not None:
            with summary_writer.as_default():
                tf.summary.scalar('total_gen_x2y_loss', avg_total_gen_x2y_loss.result(), step=epoch)
                tf.summary.scalar('total_gen_y2x_loss', avg_total_gen_y2x_loss.result(), step=epoch)
                tf.summary.scalar('disc_x_loss', avg_disc_x_loss.result(), step=epoch)
                tf.summary.scalar('disc_y_loss', avg_disc_y_loss.result(), step=epoch)

        return time.time() - start

    epoch = 0
    while True:
        execution_time = run_epoch(epoch)
        epoch += 1
        yield epoch - 1, execution_time


def plot_transformation(image_sample, generator, sample_distribution_name='X', other_distribution_name='Y',
                        fig_size=(12, 12), contrast=1, save_as=None):
    plt.figure(figsize=fig_size)

    plt.subplot(1, 2, 1)
    plt.title("Original sample from '{}'".format(sample_distribution_name))
    plt.imshow(image_sample * 0.5 * contrast + 0.5)

    plt.subplot(1, 2, 2)
    plt.title("Converted sample from to '{}'".format(other_distribution_name))
    generated_img = generator(tf.cast([image_sample], tf.float32))[0]
    plt.imshow(generated_img * 0.5 * contrast + 0.5)

    if save_as is not None:
        plt.savefig(save_as)
    plt.show()
