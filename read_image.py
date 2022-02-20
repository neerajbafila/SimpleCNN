import cv2

class Read_image:
    def __init__(self, img_path):
        self.img_path = img_path
        
    
    def read_image(self, grayScale=True):
        if grayScale:
            imgArray = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            shape_img = imgArray.shape
            return imgArray, shape_img
        else:
            imgArray = cv2.imread(self.img_path)
            imgArray = cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGB)
            shape_img = imgArray.shape
            return imgArray, shape_img
    
        