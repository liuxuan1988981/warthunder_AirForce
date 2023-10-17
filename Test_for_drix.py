from pickle import FALSE
import time
import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import keyboard as k
from win32 import win32gui
from win32 import win32api
from win32 import win32process
import win32api, win32con
import pydirectinput as direct
pyautogui.PAUSE = 0.5

def get_windows(window):
        wlist=['',' - Waiting for game',' - Loading',' - in battle',' - Test Sail']
        for state in wlist:
            windows = window + state
            handle = win32gui.FindWindow(None,windows)
            if handle == 0:
                continue
            else:
                break
        x1, y1, x2, y2 = win32gui.GetWindowRect(handle)
        return  x1, y1, x2, y2 , state

def get_son_windows(parent):
        hWnd_child_list = []
        win32gui.EnumChildWindows(parent, lambda hWnd, param: param.append(hWnd), hWnd_child_list)
        for i in hWnd_child_list:
            if win32gui.GetClassName(i) ==  'DagorWClass':
                break 
        return i


class click():
    # def __init__(self) -> None:

    def auto_Click(self,var_avg):
        pyautogui.click(var_avg[0], var_avg[1], button='left')
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
        while  True:
            x1, y1, x2, y2 ,state= get_windows('War Thunder')
            print(x1, y1, x2, y2)
            print(state)
            im = ImageGrab.grab(bbox =(x1, y1, x2, y2))
            img = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
            # img = np.array(im)#RGB
            # img = cv2.imread(im,cv2.IMREAD_COLOR)
            # imyolo = ImageGrab.grab(bbox =(x1+500, y1+200, x1+1500, y1+700))
            # imgyolo= cv2.cvtColor(np.array(imyolo), cv2.COLOR_BGR2RGB)
            # cv2.imwrite('war/' + '1' + '.jpg', imgyolo)
            # cv2.imshow('xxx',img)
            # cv2.waitKey(0)
            if state =='' or ' - Waiting for game':
                play=self.check_status("./war/to_battle.png","to_battle",img)
                if play == True:
                    self.routine("./war/to_battle.png", "to_battle",img)
                    time.sleep(5)
                else :
                    time.sleep(5)
            if state ==  '- Loading':
                time.sleep(3)
            # ingame=self.check_status("./war/ingame.png","ingame",img)
            if state == ' - in battle':
                time.sleep(1)
                if index == 0:
                    select=self.check_status("./war/select.png","select",img)
                    if select == True:
                        k.press('enter')
                        time.sleep(0.5)
                        k.release('enter')  
                        index = 1
                if index == 1:
                    ingame=self.check_status("./war/ingame.png","ingame",img)
                    if ingame == True:
                        k.press('b')
                        time.sleep(0.5)
                        k.release('b')
                        time.sleep(0.5)
                        k.press('shift')
                        time.sleep(0.5)
                        k.release('shift')
                        break
                # pyautogui.mouseDown(button='left')
                # pyautogui.mouseUp(button='left')
                
            # time.sleep(2)
def avg(x1, y1, x2, y2):
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        return x ,y
MOUSE_LEFT=0
MOUSE_MID=1
MOUSE_RIGHT=2
mouse_list_down=[win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_MIDDLEDOWN, win32con.MOUSEEVENTF_RIGHTDOWN]
mouse_list_up=[win32con.MOUSEEVENTF_LEFTUP, win32con.MOUSEEVENTF_MIDDLEUP, win32con.MOUSEEVENTF_RIGHTUP]

def mouse_down(x, y, button=MOUSE_LEFT):
    time.sleep(0.02)
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(mouse_list_down[button], 0, 0, 0, 0)

def mouse_move(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

def mouse_up(x, y, button=MOUSE_LEFT):
    time.sleep(0.02)
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(mouse_list_up[button], 0, 0, 0, 0)

def mouse_click(x, y, button=MOUSE_LEFT):
    mouse_down(x, y, button)
    mouse_up(x, y, button)

def test():
    while  True:
        x1, y1, x2, y2 ,state= get_windows('War Thunder') 
        print(state)
        # time.sleep(0.5)
        x,y=avg(x1, y1, x2, y2)
        offset=300
        # pyautogui.mouseDown(button='left')
        # pyautogui.mouseUp(button='left')
        # # pyautogui.click(x+20, y, button='right')
        # pyautogui.moveTo(x+100, y+100)
        # direct.move(offset,None)
        mouse_move(offset,0)
        time.sleep(0.5)
        time.sleep(2)
        mouse_click(offset,20)
        time.sleep(0.5)
        if k.is_pressed('esc'):
            break

    pass         

def main_run():
    ck=click()
    ck.main_gui()

if __name__ == "__main__":  
    ck=click()
    # ck.main_gui()
    test()
   

