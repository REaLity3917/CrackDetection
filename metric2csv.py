import os  
import csv  
from calculate_metrics import calculate_crack_metrics
import cv2
import pandas as pd


def read_and_process_images(image_folder, output_csv):
    # 获取图片文件列表  
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]  
      
    # 初始化结果列表，每个图片的指标将转置成列存储  
    results = []  
  
    for image_file in image_files:  

        # 读取二值化图像
        binary_image = cv2.imread(image_folder + '\\' + image_file, cv2.IMREAD_GRAYSCALE)
        binary_image = binary_image / 255
  
        # 处理图片并获取指标列表  
        metrics = calculate_crack_metrics(binary_image)  
          
        if len(results) == 0:   
            results = [[] for _ in range(len(metrics))]  
        for i, metric in enumerate(metrics):  
            results[i].append(metric)  
    
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)  
        for col in zip(*results):  
            writer.writerow(col)

    def transpose_csv(input_csv, output_csv):
        # 读取CSV文件
        df = pd.read_csv(input_csv, header=None)

        # 转置DataFrame
        df_transposed = df.T
        df_transposed.to_csv(output_csv, header=False)
        df_transposed.to_csv(output_csv, header=False, index=False)
    transpose_csv(output_csv, output_csv)


# 示例调用  
# image_folder = 'groundTruthPngImg'
# output_csv = 'metrics_samples.csv'
# read_and_process_images(image_folder, output_csv)