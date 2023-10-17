from pickle import FALSE
import time
import pyautogui
import cv2
import numpy as np
import win32api, win32con
from PIL import ImageGrab
import keyboard as k
from win32 import win32gui
from win32 import win32api
from win32 import win32process
pyautogui.PAUSE = 0.5
pyautogui.FAILSAFE=False
def mouse_move(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

def avg_get(x1, y1, x2, y2):
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        return x ,y

def get_windows(window):
        # wlist=["无",'等','载','战','试']
        wlist=['blank',' - Waiting for game',' - Loading',' - in battle',' - Test Sail']
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


class click():
    # def __init__(self) -> None:

    def auto_Click(self,var_avg):
        pyautogui.moveTo(var_avg[0], var_avg[1])  
        time.sleep(0.5)
        pyautogui.mouseDown(button='left')
        pyautogui.mouseUp(button='left')

    def get_xy(self,img_model_path,img):
        img_terminal = cv2.imread(img_model_path)
        # 读取模板的高度宽度和通道数
        height, width, channel = img_terminal.shape
        # 使用matchTemplate进行模板匹配（标准平方差匹配）
        result = cv2.matchTemplate(img, img_terminal, cv2.TM_SQDIFF_NORMED)
        # 解析出匹配区域的左上角图标
        check=cv2.minMaxLoc(result)
        if check[0]<0.1:
            upper_left = check[2]
            # 计算出匹配区域右下角图标（左上角坐标加上模板的长宽即可得到）
            lower_right = (upper_left[0] + width, upper_left[1] + height)
            # 计算坐标的平均值并将其返回
            avg = (int((upper_left[0] + lower_right[0]) / 2), int((upper_left[1] + lower_right[1]) / 2))
        else:
            avg =0
        return avg

    def check_status(self,img_model_path,name,img):
        img_terminal = cv2.imread(img_model_path)
        # 读取模板的高度宽度和通道数
        # height, width, channel = img_terminal.shape
        # 使用matchTemplate进行模板匹配（标准平方差匹配）
        result = cv2.matchTemplate(img, img_terminal, cv2.TM_SQDIFF_NORMED)
        check=cv2.minMaxLoc(result)
        if check[0]<0.1:
            status = True
            print(f"status---{name}")
        else:
            status = False
        return status    

    def routine(self,img_model_path,name,img):
        avg = self.get_xy(img_model_path,img)
        if avg == 0:
          print(f"Not have---{name}")
        else:
            print(f"clicking---{name}")
            self.auto_Click(avg)

# 'War Thunder' 'War Thunder Client'DagorWClass
    def main_gui(self):
        index=0
        wait_time =0
        while  True:
            x1, y1, x2, y2 ,state= get_windows('War Thunder')
            print(x1, y1, x2, y2)
            print(state)
            im = ImageGrab.grab(bbox =(x1, y1, x2, y2))
            img = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
            # wlist=['blank',' - Waiting for game',' - Loading',' - in battle',' - Test Sail']
            # cv2.imshow('xxx',img)
            # cv2.waitKey(0)
            if index ==2:  
                print(f"start---YOLO")
                break     
            if state =='blank' or state ==' - Waiting for game':
                play=self.check_status("./war/play.png","加入战斗",img)
                if play == True:
                    self.routine("./war/play.png", "加入战斗",img)
                    time.sleep(5)
                else :
                    if state =='blank':
                        wait_time = wait_time+1
                        print("wait_time",wait_time)
                        time.sleep(3)  
                        if wait_time >40 and index==0:
                            avgx, avgy=avg_get(x1, y1, x2, y2)
                            pyautogui.moveTo(avgx, avgy)
                            time.sleep(0.5)
                            pyautogui.mouseDown(button='left')
                            pyautogui.mouseUp(button='left')
                            print("press ---esc")
                            wait_time=0
                            k.press('esc')
                            time.sleep(0.5)
                            k.release('esc')

            if state ==  ' - Loading': 
                time.sleep(3)
            if  state == ' - in battle'or' - Test Sail':
                select=self.check_status("./war/select.png","选择",img)     
                if select == True:
                    k.press('enter')
                    time.sleep(0.5)
                    k.release('enter')  
                    time.sleep(2)
                    index = 1

                if index == 1:
                    ingame=self.check_status("./war/ingame.png","在游戏中",img)
                    if ingame == True:
                        k.press('a')
                        time.sleep(3)
                        index = 2
                        # pyautogui.mouseDown(button='right')
                        # pyautogui.mouseUp(button='right')
                        time.sleep(6)   
                        k.press('b')
                        time.sleep(0.5)
                        k.release('b')
                        time.sleep(0.5)
                        k.release('a')   
                        # k.press('shift')
                        # time.sleep(0.5)
                        # k.release('shift')       
                ingame=self.check_status("./war/ingame.png","在游戏中",img)
                if index==0 and ingame == True:
                    break
            # time.sleep(2)
def avg(x1, y1, x2, y2):
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        return x ,y
        
def test():
    while  True:
        x1, y1, x2, y2 ,state= get_windows('War Thunder') 
        print(state)
        # time.sleep(0.5)
        mouse_move(100, 0)
        time.sleep(10)
        if k.is_pressed('esc'):
            break

    pass         

def main_run():
    ck=click()
    ck.main_gui()
    return True

if __name__ == "__main__":  
    ck=click()
    ck.main_gui()
    # test()
   

