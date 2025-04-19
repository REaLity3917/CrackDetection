import cv2

# 定义鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 检测鼠标左键点击
        print(f"Clicked at pixel coordinates: ({x}, {y})")
        # 如果需要，还可以在这里添加其他处理逻辑，例如获取该点的颜色等

# 读取图像
image = cv2.imread('configs//test.jpg')

# 检查图像是否成功加载
if image is None:
    print("Error: Unable to load image.")
    exit()

# 获取图像的原始尺寸
height, width = image.shape[:2]

# 计算缩放比例
scale = min(720 / height, 1280 / width)

# 缩放图像
resized_image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

# 转换为灰度图像
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测
edges = cv2.Canny(gray, 50, 150)

# 寻找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓并绘制边界框
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(x, y, w, h)

# 创建窗口并设置鼠标回调函数
cv2.namedWindow('Image with bounding boxes')
cv2.setMouseCallback('Image with bounding boxes', mouse_callback)

# 显示带有边界框的图像
cv2.imshow('Image with bounding boxes', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()