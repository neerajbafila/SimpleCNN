from matplotlib import pyplot as plt
import tensorflow as tf
from reShape_image import Reshape_image
class CNN_with_pooling:
    def __init__(self, imgArray):
        self.imgArray = imgArray
        self.reshape_ob = Reshape_image(self.imgArray)
    
    def model_with_pooling(self, pool_size = (2,2,), strides=(2,2)):
        pooling_layer = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides)
        new_img, shape = self.reshape_ob.reShape_image_for_cnn_pred()
        print(f'shape of new img is {shape}')
        result = pooling_layer(new_img)
        return result
    
    def model_with_global_avg_pooling(self):
        pooling_layer = tf.keras.layers.GlobalAvgPool2D()
        newImg, shape = self.reshape_ob.reShape_image_for_cnn_pred()
        print(f'shape of new img is {shape}')
        result_avg = pooling_layer(newImg)
        return result_avg

    
    def plot_pooling(self, pooling_result):
        print(pooling_result.shape)
        _, row, col, _ = pooling_result.shape
        reshape_out = tf.reshape(pooling_result, (row, col, -1))
        plt.imshow(reshape_out, cmap="gray")
        plt.show()
        print(f"shape of reshape_out is {reshape_out.shape}")

    def plot_pooling_global(self, pooling_result):
        print(pooling_result.shape)
        row, col= pooling_result.shape
        reshape_out = tf.reshape(pooling_result, (row, col, -1))
        plt.imshow(reshape_out, cmap="gray")
        plt.show()
        print(f"shape of reshape_out is {reshape_out.shape}")