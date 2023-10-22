import cv2
import pytesseract

# 读取图像
image = cv2.imread('findway/image.png')

# 图像预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

# 文字识别
text = pytesseract.image_to_string(denoised, lang='eng')

# 输出识别结果

print(text)