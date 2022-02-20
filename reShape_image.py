from typing_extensions import Self

class Reshape_image:
    def __init__(self, imgArray):
        self.imgArray = imgArray
    
    def reShape_image_for_cnn_inp(self):
        row, col, depth = self.imgArray.shape
        new_imgArray = self.imgArray.reshape(row,col,depth) 
        return new_imgArray, new_imgArray.shape
    
    def reShape_image_for_cnn_pred(self):
        row, col, depth = self.imgArray.shape
        new_imgArray = self.imgArray.reshape(1,row,col,depth) 
        return new_imgArray, new_imgArray.shape

