# -*- coding: utf-8 -*-
"""TF 2.0 Fast-SCNN.ipynb
"""

# !pip install tensorflow-gpu==2.0.0-alpha0

import tensorflow as tf

"""
# Model Architecture
#### Custom function for conv2d: conv_block
"""
def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
  
  if(conv_type == 'ds'):
    x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)  
  
  x = tf.keras.layers.BatchNormalization()(x)
  
  if (relu):
    x = tf.keras.activations.relu(x)
  
  return x

"""## Step 2: Global Feature Extractor

#### residual custom method
"""

def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    
    
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method"""

def bottleneck_block(inputs, filters, kernel, t, strides, n):
  x = _res_bottleneck(inputs, filters, kernel, t, strides)
  
  for i in range(1, n):
    x = _res_bottleneck(x, filters, kernel, t, 1, True)

  return x

"""#### PPM Method"""

def pyramid_pooling_block(input_tensor, bin_sizes):
  height = int(input_tensor.shape[1])
  width = int(input_tensor.shape[2])
  concat_list = [input_tensor]

  for bin_size in bin_sizes:
    if height % bin_size != 0 or width % bin_size != 0:
      raise ValueError(
        f'bin_size={bin_size} deve dividir o feature map {height}x{width}.'
      )

    pool_height = height // bin_size
    pool_width = width // bin_size

    x = tf.keras.layers.AveragePooling2D(
      pool_size=(pool_height, pool_width),
      strides=(pool_height, pool_width),
      padding='valid'
    )(input_tensor)
    x = tf.keras.layers.Conv2D(128, (1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Resizing(height, width, interpolation='bilinear')(x)

    concat_list.append(x)

  return tf.keras.layers.concatenate(concat_list)


def build_fast_scnn(num_classes, input_shape=(320, 320, 3), pyramid_bins=(1, 2, 5, 10)):
  """Cria uma Fast-SCNN não compilada e retorna logits por pixel."""
  height, width, channels = input_shape

  if channels != 3:
    raise ValueError('A Fast-SCNN espera imagens RGB com 3 canais.')
  if height % 32 != 0 or width % 32 != 0:
    raise ValueError('A altura e a largura da entrada devem ser divisíveis por 32.')

  input_layer = tf.keras.layers.Input(shape=input_shape, name='input')

  # Step 1: Learning to Downsample
  lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides=(2, 2))
  lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides=(2, 2))
  lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides=(2, 2))

  # Step 2: Global Feature Extractor
  gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
  gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
  gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
  gfe_layer = pyramid_pooling_block(gfe_layer, pyramid_bins)

  # Step 3: Feature Fusion
  ff_layer1 = conv_block(
    lds_layer, 'conv', 128, (1, 1), padding='same', strides=(1, 1), relu=False
  )

  ff_layer2 = tf.keras.layers.UpSampling2D(
    (4, 4), interpolation='bilinear'
  )(gfe_layer)
  ff_layer2 = tf.keras.layers.SeparableConv2D(
    128,
    (3, 3),
    padding='same',
    strides=(1, 1),
    activation=None,
    dilation_rate=(4, 4),
    use_bias=False
  )(ff_layer2)
  ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
  ff_layer2 = tf.keras.activations.relu(ff_layer2)
  ff_layer2 = tf.keras.layers.Conv2D(
    128, (1, 1), strides=(1, 1), padding='same', activation=None, use_bias=False
  )(ff_layer2)

  ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
  ff_final = tf.keras.layers.BatchNormalization()(ff_final)
  ff_final = tf.keras.activations.relu(ff_final)

  # Step 4: Classifier
  classifier = tf.keras.layers.SeparableConv2D(
    128, (3, 3), padding='same', strides=(1, 1), name='DSConv1_classifier'
  )(ff_final)
  classifier = tf.keras.layers.BatchNormalization()(classifier)
  classifier = tf.keras.activations.relu(classifier)

  classifier = tf.keras.layers.SeparableConv2D(
    128, (3, 3), padding='same', strides=(1, 1), name='DSConv2_classifier'
  )(classifier)
  classifier = tf.keras.layers.BatchNormalization()(classifier)
  classifier = tf.keras.activations.relu(classifier)

  classifier = tf.keras.layers.Conv2D(
    num_classes, (1, 1), strides=(1, 1), padding='same', name='classifier_logits'
  )(classifier)
  classifier = tf.keras.layers.Dropout(0.3)(classifier)

  logits = tf.keras.layers.UpSampling2D(
    (8, 8), interpolation='bilinear', name='logits'
  )(classifier)

  return tf.keras.Model(inputs=input_layer, outputs=logits, name='Fast_SCNN')
