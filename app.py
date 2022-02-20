
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
from read_image import Read_image
from one_filter_cnn_model import One_filter_cnn_model
from reShape_image import Reshape_image
from plot_result_of_cnn_pred import Plot_result_of_cnn_pred


img_read = Read_image('car1.jpeg')
imgArray, shape = img_read.read_image(grayScale=False)
reshape_ob = Reshape_image(imgArray)
new_imgArray, shape = reshape_ob.reShape_image_for_cnn_inp()
model_one_filter = One_filter_cnn_model(shape)
model = model_one_filter.model(filter=5)




# shape_for_cnn = (shape[0], shape[1], 1)
# print(shape_for_cnn)
# model_one_filter = One_filter_cnn_model(shape_for_cnn)
# model = model_one_filter.model()
# row, col = shape
# imgArray = imgArray.reshape(1,row,col,1)
# print(imgArray.shape)
try:
    new_imgArray_pred, shape = reshape_ob.reShape_image_for_cnn_pred()
    out = model.predict(new_imgArray_pred)
except Exception as e:
    print(e)

ob_plot_pred = Plot_result_of_cnn_pred(out)
ob_plot_pred.plot_result()


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
