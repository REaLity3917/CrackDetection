#!/user/bin/python
# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, optimizer
import torchvision
import os, sys
import cv2 as cv
from torch.utils.data import DataLoader, sampler
from train.train import UNetModel

def test(unet, root_dir, results_dir,current_file = None):
    model_dict=unet.load_state_dict(torch.load('train\\unet_road_model.pt'))
    # root_dir = 'test/'
    # results_dir = 'results/'
    if current_file:
        fileNames = [current_file]
    else:
        fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f), cv.IMREAD_GRAYSCALE)

        #image = cv.resize(image, (480, 320))

        h, w = image.shape
        img = np.float32(image) /255.0
        img = np.expand_dims(img, 0)
        x_input = torch.from_numpy(img).view( 1, 1, h, w)
        probs = unet(x_input.cuda())
        m_label_out_ = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        grad, output = m_label_out_.data.max(dim=1)
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()

        # print(predic_)
        # print(predic_.max())
        # print(predic_.min())

        # print(predic_)
        # print(predic_.shape)
        # cv.imshow("input", image)
        result = cv.resize(np.uint8(predic_), (w, h))

        # 将原图和分割结果拼接在一起
        # combined = cv.hconcat([image, result])
                # 保存拼接后的图像到results文件夹
        # result_filename = os.path.join(results_dir, f"combined_{f}")
        # cv.imwrite(result_filename, combined)
        result_filename = os.path.join(results_dir, f"{f}")
        cv.imwrite(result_filename, result)
        # 显示拼接后的图像
        # cv.imshow("Original and Segmented", combined)
        # cv.waitKey(0)
    # cv.destroyAllWindows()
# if __name__ == "__main__":
    unet = UNetModel().cuda()
#     test(unet)  # 调用测试函数