import cv2
import pytesseract

def ocr_read(image):
    # 图像预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    # 文字识别
    text = pytesseract.image_to_string(denoised, lang='eng')
    return text
# 输出识别结果

def ocr_read_gray(image):
    # 图像预处理
    # print("muther fuck")
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    _, binary = cv2.threshold(image[..., 2], 230, 255, cv2.THRESH_BINARY_INV)
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    # 腐蚀一下
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    dst=cv2.erode(denoised,kernel)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 文字识别
    text = pytesseract.image_to_string(dst, lang='eng')
    return text


if __name__ == "__main__":  

    image = cv2.imread("./123.png")
    # (x1+25, y1+48, x1+275, y1+120)
    # x1=0
    # y1=0
    # cropped_image =image[y1+48:y1+120, x1+25:x1+275]
    output=ocr_read_gray(image)
    print(output)
    index = output.find("SPD") and output.find("ALT")
    if index != -1:
    # 提取出 "SPD" 后面的数字
        speed = output.split("SPD ")[1].split(" ")[0]
        print(int(speed),"111")
        alt = output.split("ALT ")[1].split(" ")[0]
        print(alt,"222")
    cv2.imshow('Cropped Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()