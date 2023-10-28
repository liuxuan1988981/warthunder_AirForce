import cv2
import pytesseract

def ocr_read(image):
    # 图像预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    # 文字识别
    text = pytesseract.image_to_string(denoised, lang='eng')
    return text
# 输出识别结果



if __name__ == "__main__":  

    image = cv2.imread('findway/image.png')
    # (x1+25, y1+48, x1+275, y1+120)
    x1=0
    y1=0
    cropped_image =image[y1+48:y1+120, x1+25:x1+275]
    output=ocr_read(cropped_image)
    print(output)
    index = output.find("SPD") and output.find("ALT")
    if index != -1:
    # 提取出 "SPD" 后面的数字
        speed = output.split("SPD ")[1].split(" ")[0]
        print(int(speed),"111")
        alt = output.split("ALT ")[1].split(" ")[0]
        print(alt,"222")
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()