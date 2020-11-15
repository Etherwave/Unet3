import os
import cv2
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader
from Data.MyDataSet import Test_DataSet
from models.MyModel import UNet_3Plus

# torch.set_num_threads(1)
model_save_path = "../save/unet3.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    model = UNet_3Plus()

    # load model
    model.load_state_dict(torch.load(model_save_path,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # load image data
    batch_size = 1
    num_workers = 1
    dataset = Test_DataSet()
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, sampler=None)

    for i, data in enumerate(test_loader):
        image = data[0]
        label = data[1]
        data = {
            'image':           image.to(device),
            'label':           label.to(device),
        }

        outputs = model.test_forword(data)
        feature_map = outputs['feature_map']
        loss = outputs['total_loss']
        print("loss is {}".format(loss))

        image = image.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        feature_map = feature_map.cpu().detach().numpy()

        feature_map = feature_map*255

        image = np.uint8(image)
        label = np.uint8(label)
        feature_map = np.uint8(feature_map)

        image = image.squeeze()
        image = np.transpose(image, (1, 2 , 0))
        label = label.squeeze()
        label = np.expand_dims(label, -1)
        label*=255
        feature_map = feature_map.squeeze()
        feature_map = np.expand_dims(feature_map, -1)

        # print(image.shape)
        # print(label.shape)
        # print(feature_map.shape)

        cv2.imshow("image", image)
        cv2.imshow("label", label)
        cv2.imshow("feature_map", feature_map)
        cv2.waitKey()

if __name__ == '__main__':
    main()
