import matplotlib.pyplot as plt

class Plot_result_of_cnn_pred:
    def __init__(self,cnn_outcome):
        self.cnn_img = cnn_outcome
    
    def plot_result(self):
        batch, r, c, d = self.cnn_img.shape
        # print(d)
        # print(f"{batch},{r}, {c}, {d}")
        if d < 1:
            new_img = self.cnn_img.reshape(r,c,d)
            plt.imshow(new_img, cmap='gray')
            plt.show()
        else: 
            new_img = self.cnn_img.reshape(r,c,d)
            for i in range(d):
                plt.imshow(new_img[:,:,i], cmap='gray')
                plt.show()
                if i == 10: # only display 10 pic
                    break

            # if counter <= d_list[-2]:
            #     new_img = self.cnn_img.reshape(r,c,d)
            #         # img = new_img.reshape(r,c,d_list[counter])
            #     plt.imshow(new_img[:,:,d_list[counter]])
            #     # plt.imshow(new_img[:,:,i], cmap='gray')
            #         # plt.show(new_img[:,:,d[c]])
            #     plt.show()
            #     counter=counter+1
