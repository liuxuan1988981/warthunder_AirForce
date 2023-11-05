# coding:utf-8
# 多进程
import threading
import time
from multiprocessing import Process, Manager
import operator
from cv2 import imshow
from ProcessYolo import run as yolo_run
from ProcessYolo import init_func as yolo_init
from autoclick import main_run
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
from findway.ocr import ocr_read_gray
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
pyautogui.PAUSE = 0.5
pyautogui.FAILSAFE=False
MOUSE_LEFT=0
MOUSE_MID=1
MOUSE_RIGHT=2
mouse_list_down=[win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_MIDDLEDOWN, win32con.MOUSEEVENTF_RIGHTDOWN]
mouse_list_up=[win32con.MOUSEEVENTF_LEFTUP, win32con.MOUSEEVENTF_MIDDLEUP, win32con.MOUSEEVENTF_RIGHTUP]
wait_timer=0



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

def mouse_moveclick(x, y, button=MOUSE_LEFT):
    mouse_down(x, y, button)
    mouse_up(x, y, button)

def mouse_click(button=MOUSE_LEFT):
    time.sleep(0.02)
    win32api.mouse_event(mouse_list_up[button], 0, 0, 0, 0)
    time.sleep(0.02)
    win32api.mouse_event(mouse_list_down[button], 0, 0, 0, 0)

def list_equal (a,b):
    if len(a)!=len(b):
        return False
    for i in range (len(a)):
        if a[i]!= b[i]:
            return False
    return True
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


def avg_get(x1, y1, x2, y2):
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        return x ,y
def pixel_to_int(px,py):
    ix=px*2.2
    iy=py*2.2
    return int(ix),int(iy)
def check_status(img_model_path,name,img):
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

def check_AIRstatus(x1,y1,x2, y2):
        # this is rgb channel
        img = ImageGrab.grab(bbox =(x1+25, y1+50, x1+300, y1+250))
        # get r channel
        # image = np.array(img)
        image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)  
        # red_img,g,b = img[:,:,0], img[...,1], img[...,2] 
        text=ocr_read_gray(image)
        if "SPD" or "ALT" in text:
            status = True
            print("check S A OK")
        else:
            status = False
            print(" S A NULL")
        return status     


class ProcessGui():
    '''
    主函数
    '''    
    def __init__(self):
        # super().__init__()
        # self.queuePut = queuePut
        # self.queueGet = queueGet   
        pass 
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

    def routine(self,img_model_path,name,img):
        avg = self.get_xy(img_model_path,img)
        if avg == 0:
          print(f"Not have---{name}")
        else:
            print(f"clicking---{name}")
            self.auto_Click(avg)

    def routine_direx(self,img_model_path,name,img):
        avg = self.get_xy(img_model_path,img)
        if avg == 0:
          print(f"Not have---{name}")
        else:
            print(f"clicking---{name}")           
            direct.moveTo(avg[0], avg[1]+10)
            time.sleep(0.5)
            pyautogui.mouseDown(button='left')
            time.sleep(0.5)
            pyautogui.mouseUp(button='left') 
   
    def run(self):         
        while True:
            if k.is_pressed('esc'):
                break 
            x1, y1, x2, y2 ,state= get_windows('War Thunder')
            print(x1, y1, x2, y2)
            print(state)
            start_falg=False

            Gamingcheck=check_AIRstatus(x1, y1, x2, y2)
            
            if Gamingcheck==True:
                start_falg=True
            else:
                start_falg=main_run()
            time.sleep(0.5)
         # 开始处理数据
            dead_flag=False
            ########ProcessYolo######################
            if start_falg==True:
                start_thread()
            else :
                break
            #########################################
            check_false_times=0

            while start_falg==True:        
                if k.is_pressed('esc'):
                    break
                # todo:这里处理界面信息
                time.sleep(5)

                x1, y1, x2, y2 ,state= get_windows('War Thunder')
                avgx,avgy = avg_get(x1, y1, x2, y2)
                print(state)
                print(x1, y1, x2, y2)
                img = ImageGrab.grab(bbox =(x1, y1, x2, y2))
                img_check = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)               
                # imyolo = ImageGrab.grab(bbox =(x1+250, y1+100, x1+1500, y1+1000))
                # imyolo = ImageGrab.grab(bbox =(x1+200, y1+100, x1+1800, y1+900))
                # imgyolo= cv2.cvtColor(np.array(imyolo), cv2.COLOR_BGR2RGB)
                # imgyolo = cv2.imread(imyolo,cv2.IMREAD_COLOR)
                # imgyolo = imyolo        
                # 测试数据                  
                # add ocr function to check aleart and speed 
                firstcheck=check_AIRstatus(x1, y1, x2, y2)
                if firstcheck == True:
                    print("====== In gaming ======")
                    battle_falg=True  
                    if dead_flag==True and battle_falg==True:
                        dead_flag=False
                        start_thread()
                        print("!!start again!!")
                    continue      
                else:   
                    ########Stop Yolo################
                    print("!!!!!!!~~~~ Stop Yolo~~~~~~~~!!!!!!!")
                    stop_thread()
                    ################################
                    time.sleep(1)             
                if  state == ' - in battle' and firstcheck == False:
                    # mouse_move(avgx, avgy)
                    dead=check_status("./war/dead.png","dead",img_check)
                    if dead==True:
                        k.press('enter')
                        time.sleep(0.5)
                        k.release('enter') 
                        # time.sleep(4)          
                        dead_flag=True                
                    return_base=check_status("./war/return_base.png","return_base",img_check)
                    if return_base==True:
                        self.routine_direx("./war/return_base.png","return_base",img_check)  
                         
                if state =='blank' :
                    time.sleep(0.5)
                    direct.moveTo(avgx, avgy)
                    time.sleep(0.5)
                    pyautogui.mouseDown(button='left')
                    pyautogui.mouseUp(button='left')
                    time.sleep(3.5)
                    k.press('enter')
                    time.sleep(0.5)
                    k.release('enter')   
                    break
            pass
              ##ProcessYolo while
        pass
          #while
        
class ProcessYolo(Process):
    '''
    yolo进程
    '''
    def __init__(self, queueGet, queuePut):
        super().__init__()
        self.queueGet = queueGet
        self.queuePut = queuePut  

    def serch_the_ship(self):
        print("====== time > 100 search the  ship ======")  
        # k.press('space')
        # time.sleep(0.5)        
        offset=3000
        mouse_move(offset,0)
        time.sleep(0.5)
        # direct.move(offset,None)
    
    def Check_SPD_ALT (self,x1,y1,x2, y2):
        speed=0
        alt=0
        img = ImageGrab.grab(bbox =(x1+25, y1+48, x1+500, y1+250))
        image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        txt=ocr_read_gray(image)
        # print(txt,"SPEED-ALT")
        index = txt.find("SPD") and txt.find("ALT")
        if index != -1:
        # 提取出 "SPD" 后面的数字
            speed_char = txt.split("SPD ")[1].split(" ")[0]
            # print(speed,"speed char")
            speed_char = speed_char.replace("O", "0").replace("i", "1")
            speed = int(speed_char)
            print(speed,"SPD Num ")
            alt_char = txt.split("ALT ")[1].split("m")[0]            
            alt_char = alt_char.replace("O", "0").replace("i", "1")
            # print(alt,"speed char")
            alt = int(alt_char)
            print(alt,"ALT Num")
        return speed,alt

    def Take_Off_Procedure(self,ground,x1, y1, x2, y2):
        print("====== Take off ======")
        k.press('w')
        time.sleep(3)
        k.release('w') 
        while True:
            spd,alt=self.Check_SPD_ALT(x1,y1,x2,y2)
            if alt-ground<10:     
                if spd > 230 :
                    mouse_move(0,-1000)
                    print("====== Take off move(0,-1000)======")
                    time.sleep(4)                   
            else:
                if alt > ground+800:
                    print("====== Eagle is Flying ======")
                    break
            time.sleep(1)
        return alt
    
    def ALTITUDE_Control_Procedure(self,x1, y1, x2, y2):       
        spd,alt=self.Check_SPD_ALT(x1,y1,x2,y2)
        if alt-800>10:      
            mouse_move(0,50)
        else:
            mouse_move(0,-50)

    def Bombing_Procedure(self,ground,ix, iy,x1, y1, x2, y2):   
        while True:
            spd,alt=self.Check_SPD_ALT(x1,y1,x2,y2)
            if alt-ground>350:      
                mouse_move(ix,iy)
            else:
                k.press('ctrl')
                time.sleep(0.5)
                k.release('ctrl') 
                mouse_move(0,1000)
                time.sleep(0.5)
                k.press('w')
                time.sleep(2)
                k.release('w') 
                break
        return True

    def run(self):
        # 首先初始化
        # preflight station ckeck
        print("====== YOLO Start ======")
        x1, y1, x2, y2 ,state= get_windows('War Thunder')
        avgx,avgy = avg_get(x1, y1, x2, y2)
        testfire=0
        trun_r=0
        fire_times =0
        not_ship_time=0
        not_targe_time=0
        ALTITUDE=0
        ALTITUDE_Ground = 0
        SPEED,altnum=self.Check_SPD_ALT(x1, y1, x2, y2)
        
        if SPEED==0:
            ALTITUDE_Ground=altnum 
            #  take off
            ALTITUDE=self.Take_Off_Procedure(ALTITUDE_Ground,x1, y1, x2, y2)
        print("====== initialize ======")                         
        yolo_init()
        print("====== YOLO initialize complete ======")                  
        while  True:
            if k.is_pressed('esc'):
                break
            imyolo = ImageGrab.grab(bbox =(x1, y1+100, x1+1600, y1+900))
            imgyolo= cv2.cvtColor(np.array(imyolo), cv2.COLOR_BGR2RGB)  
            label ,boxes=yolo_run(imgyolo)
            # print(label ,boxes)         
            not_targe_time= not_targe_time+1
            # if len(label) != 0 and (not list_equal([1],label)) and (not list_equal([1,2],label)) and (not list_equal([2],label)) :
            if len(label) != 0:
                print("====== Bomb Point in Scream ======")
                dx, dy=avg_get(boxes[len(label)-1][0]*1600+x1,boxes[len(label)-1][1]*800+y1,boxes[len(label)-1][2]*1600+x1,boxes[len(label)-1][3]*800+y1)   
                ix, iy=pixel_to_int(dx-avgx,dy-avgy)
                if iy<700:
                    mouse_move(ix, 0)
                    time.sleep(0.5)  
                    self.ALTITUDE_Control_Procedure(x1, y1, x2, y2)
                else: 
                    print("====== Start Bombing======")
                    k.press('s')
                    time.sleep(1.5)
                    k.release('s')
                    Bomb_flag=self.Bombing_Procedure(ALTITUDE_Ground,ix, iy,x1, y1, x2, y2)
   
            else:
                self.ALTITUDE_Control_Procedure(x1, y1, x2, y2) 
      
        pass   


def start_thread():
    # 创建并启动进程
    global my_process
    qGuiGet = Manager().Queue()
    qYoloGet = Manager().Queue()
    my_process = ProcessYolo(qYoloGet, qGuiGet)
    my_process.start()

def stop_thread():
    # 停止进程
    my_process.terminate()
    my_process.join()  # 等待进程结束

if __name__ == "__main__":
    ps=ProcessGui()
    ps.run()    
    
    # qGuiGet = Manager().Queue()
    # qYoloGet = Manager().Queue()
    # pGui = ProcessGui(qGuiGet, qYoloGet)
    # pYolo = ProcessYolo(qYoloGet, qGuiGet)
    # pGui.start()
    # pYolo.start()
    # pGui.join()
    # pYolo.join()
