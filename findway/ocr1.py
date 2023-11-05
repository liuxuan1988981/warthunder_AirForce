# coding:utf-8
# ocr识别
import cv2
import pytesseract

# 读取图像
# image = cv2.imread(r'C:\aaaa.png')
# image = cv2.imread(r'C:\bbb.png')
# image = cv2.imread(r'C:\111.png')
# image = cv2.imread(r'C:\222.png')
# image = cv2.imread(r'C:\333.png')
image = cv2.imread("./123.png")
cv2.imshow("img", image)
# 图像预处理
_, binary = cv2.threshold(image[..., 2], 230, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("binary", binary)
denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
cv2.imshow("denoise", denoised)
# 腐蚀一下
kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
dst=cv2.erode(denoised,kernel)
cv2.imshow("erode_demo",dst)
# 文字识别
text = pytesseract.image_to_string(dst, lang='eng')


def ocr_read_gray(image):
    # 图像预处理
    print("muther fuck")
    _, binary = cv2.threshold(image[..., 2], 230, 255, cv2.THRESH_BINARY_INV)
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    # 腐蚀一下
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    dst=cv2.erode(denoised,kernel)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 文字识别
    text = pytesseract.image_to_string(dst, lang='eng')
    return text


# 输出识别结果
print("===")
print(text)
print("===")
cv2.waitKey(0)
