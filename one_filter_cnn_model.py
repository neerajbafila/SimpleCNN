import tensorflow as tf

class One_filter_cnn_model:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def model(self, filter=1):
        LAYER = [tf.keras.layers.Conv2D(filters=filter, kernel_size=(3,3), strides=1, padding='VALID', input_shape=self.input_shape)]
        conv_model = tf.keras.Sequential(LAYER)
        conv_model.summary()
        return conv_model
    
