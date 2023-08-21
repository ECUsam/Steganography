import cv2

# 读取BMP格式的图像文件
img = cv2.imread('lena_CONV_1.bmp')

# 显示图像
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()