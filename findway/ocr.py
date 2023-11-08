import cv2
import pytesseract
import threading
import time
from multiprocessing import Process, Manager
import operator
from cv2 import imshow
import pydirectinput as direct
import cv2
import time
import pyautogui
import numpy as np
from PIL import ImageGrab
import keyboard as k
from win32 import win32gui
from win32 import win32api
from win32 import win32process
import win32api, win32con
import os 

def get_windows(window):
        # wlist=["无",'等','载','战','试']
        wlist=['blank',' - Waiting for game',' - Loading',' - In battle',' - Test Flight']
        index=0
        while True: 
            window_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            for state in wlist: 
                if window_name == window:
                    handle = win32gui.FindWindow(None,window_name)
                    if handle == 0:
                        continue
                    else:
                        state="blank"
                        index=1
                        break
                if state in window_name:
                    handle = win32gui.FindWindow(None,window_name)
                    if handle == 0:
                        continue
                    else:
                        index=1
                        break
            if index==1:
                break
        x1, y1, x2, y2 = win32gui.GetWindowRect(handle)
        return  x1, y1, x2, y2 , state

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

def Check_SPD_ALT (x1,y1,x2, y2):
        speed=0
        alt=0
        img = ImageGrab.grab(bbox =(x1+25, y1+80, x1+275, y1+150))
        image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        txt=ocr_read_gray(image)
        # print(txt,"output SPEED-ALT~~~~~~~~~~~~")
        index = (txt.find("SPD")) * (txt.find("ALT"))
        print(index,"index")
        if index ==0 or index >1 :
        # 提取出 "SPD" 后面的数字  
            try:  
                result=True
                speed_char = txt.split("SPD ")[1].split(" ")[0]
                # print(speed_char,"speed char")
                speed_char = speed_char.replace("O", "0").replace("i", "1")
                speed = int(speed_char)
                print(speed,"SPD Num ")
                alt_char = txt.split("ALT ")[1].split("m")[0]            
                alt_char = alt_char.replace("O", "0").replace("i", "1")
                # print(alt_char,"alt_char")
                alt = int(alt_char)
                print(alt,"ALT Num")
            except IndexError: 
                print("IndexError") 
                result=False
            except ValueError : 
                print("ValueError") 
                result=False      
        else:
            result=False
        return  result, speed,alt

if __name__ == "__main__":  

    x1, y1, x2, y2 ,state= get_windows('War Thunder')
    Check_SPD_ALT(x1, y1, x2, y2)

    
    # cv2.imshow('Cropped Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()