# 使用方法by JaspinXu

这个只能用来作为可以计算分数的demo，算法就是指标+熵权法+模糊综合评价，非常非常基础没有一点技术含量，只能起应付中期验收的作用，实际用还得改，只能说是程序能跑的程度......

## dataset

数据集存放在groundTruthPngImg目录下，用来产生权重

## calculate_metrics.py

这个文件用来计算各种指标，但是有的指标稍微有点重复，指标越多越好，其中最后的index_basic_metric是预先设定权重的，主观性很大，需要根据情况修改

## metric2csv.py

运行这个文件会计算数据集中所有图像的所有指标，然后存到metrics_samples.csv文件里，其中一列代表一个图像的22个指标，一行代表83张数据集图像的同一指标计算值

## crack_entropy_fuzzy.py

运行这个文件会由csv文件计算一些权重，对于新的图像，比如crack_binary_image.png，计算它的所有指标值，然后计算隶属度矩阵(这个地方偏主观，等级向量level_vectors各个临界值的确定需要依据指标对损伤程度的主观判断)，再利用上面的权重计算出综合得分，分数越低损伤程度越大

## 路径修改

metric2csv.py的image_folder = 'G:\\crack_estimate\\crack_estimate\\groundTruthPngImg' 用的时候改成自己的路径

## pipeline

可以修改groundTruthPngImg目录下的数据集，运行metric2csv.py，注意生成的csv文件中可能会出现数值问题，比如存在126.1962166.1这种数据，需要手动把后面的.1去除，改成126.1962166，然后对于新的裂缝图片，替换crack_entropy_fuzzy.py文件里的binary_image = cv2.imread('crack_binary_image.png', cv2.IMREAD_GRAYSCALE)这里的png图片，然后运行crack_entropy_fuzzy.py即可得出分数

## 需要改进的地方

计算更多指标，筛除多余的无用指标

一些系数的确定偏主观，可以再优化

隶属度矩阵等级向量level_vectors需要根据指标具体分析进行修改

主观指标index_basic_metric = calculate_basic_metrics(orign_binary_image,0.25,0.25,0.25,0.25)要改后边的权重，默认用的全是0.25，必然不行，因为这会导致裂缝数量的影响微乎其微

改指标时is_benefit和level_vectors都要修改

代码中一堆数据类型转换，比如astype，极大地增加了误差

基本是torch的tensor和numpy的array格式都在用，建议全改成tensor格式，然后全移到gpu里去存算


