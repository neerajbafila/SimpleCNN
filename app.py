
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
from read_image import Read_image
from one_filter_cnn_model import One_filter_cnn_model
from reShape_image import Reshape_image
from plot_result_of_cnn_pred import Plot_result_of_cnn_pred
from cnn_with_pooling import CNN_with_pooling


# for cnn with filters
# img_read = Read_image('car1.jpeg')
# # img_read = Read_image('car_gray.jpeg')
# imgArray, shape = img_read.read_image(grayScale=False)
# reshape_ob = Reshape_image(imgArray)
# new_imgArray, shape = reshape_ob.reShape_image_for_cnn_inp()
# model_one_filter = One_filter_cnn_model(shape)
# model = model_one_filter.model(filter=3)
# try:
#     new_imgArray_pred, shape = reshape_ob.reShape_image_for_cnn_pred()
#     out = model.predict(new_imgArray_pred)
# except Exception as e:
#     print(e)

# ob_plot_pred = Plot_result_of_cnn_pred(out)
# ob_plot_pred.plot_result()


# for CNN with only pooling
img_read = Read_image('car1.jpeg')
# img_read = Read_image('car_gray.jpeg')
imgArray, shape = img_read.read_image(grayScale=False)
ob_pooling = CNN_with_pooling(imgArray)
result1 = ob_pooling.model_with_pooling(pool_size=(2,2), strides=(2,2))
ob_pooling.plot_pooling(result1)
# for globalAvgPooling
ob_pooling = CNN_with_pooling(imgArray)
result2 = ob_pooling.model_with_global_avg_pooling()
ob_pooling.plot_pooling_global(result2)






# shape_for_cnn = (shape[0], shape[1], 1)
# print(shape_for_cnn)
# model_one_filter = One_filter_cnn_model(shape_for_cnn)
# model = model_one_filter.model()
# row, col = shape
# imgArray = imgArray.reshape(1,row,col,1)
# print(imgArray.shape)
# print(out.shape)
# row, col = out.shape[1:-1]
# reshape_out = out.reshape(row, col)
# print(f"reshape is {reshape_out}")
# plt.imshow(reshape_out, cmap="gray")
# plt.show()

#For gray Scale Image
# ob_read = Read_image('car1.jpeg')
# arr, shape = ob_read.read_image(grayScale=True)
# print(f"imgArray is {arr}, shape is {shape}")
# plt.imshow(arr, cmap='gray')
# plt.show()

#For gray color Image
# ob_read = Read_image('car1.jpeg')
# arr, shape = ob_read.read_image(grayScale=False)
# print(f"imgArray is {arr}, shape is {shape}")
# plt.imshow(arr)
# plt.show()
# row, col, depth = shape
# print(f"row is {row}, col is {col}, depth is {depth}")
