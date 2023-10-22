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
            start_falg=False
            if k.is_pressed('esc'):
                break
            
            start_falg=main_run()
            time.sleep(0.5)
         # 开始处理数据
            dead_flag=False
            #########ProcessYolo######################
            if start_falg==True:
                start_thread()
            else:
                break
            ##########################################
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
                #                  

                # add ocr function to check aleart and speed 
                firstcheck=check_status("./war/ingame.png","在游戏中",img_check)

                if firstcheck == True:
                    print("====== In gaming ======")
                    battle_falg=True  
                    time.sleep(2)   
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
                    # State_Detection(state,img_check,x1, y1, x2, y2)  
                    # print("!!!!!!!~~~~ State_Detection~~~~~~~~!!!!!!!")
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
        fire_times=0
        if trun_r <5:
            trun_r=trun_r+1
            offset=3000
        else :
            offset =-4000
        mouse_move(offset,0)
        time.sleep(0.5)
        # direct.move(offset,None)
        direct.press('l')
        time.sleep(0.5)
        # k.release('space')
        direct.press('c') 
    
    def run(self):
        # 首先初始化
        # label_list=["light","target","battel","ship","destroy"] 
        print("====== initialize ======")     
        k.press('shift')
        time.sleep(0.5)
        k.release('shift')                         
        yolo_init()
        k.press('b')
        # 发送初始化完成的信号
        print("====== YOLO initialize complete ======")
        x1, y1, x2, y2 ,state= get_windows('War Thunder')
        avgx,avgy = avg_get(x1, y1, x2, y2)
        testfire=0
        trun_r=0
        fire_times =0
        not_ship_time=0
        not_targe_time=0
        time.sleep(0.5)
        k.release('b')
        time.sleep(0.5)
        pyautogui.mouseDown(button='left')
        pyautogui.mouseUp(button='left')
        direct.press('l')
        time.sleep(0.5)
        while  True:
            if k.is_pressed('esc'):
                break
            imyolo = ImageGrab.grab(bbox =(x1, y1+100, x1+1600, y1+900))
            imgyolo= cv2.cvtColor(np.array(imyolo), cv2.COLOR_BGR2RGB)
            # cv2.imshow('img', img)
            # cv2.waitKey(1)
            label ,boxes=yolo_run(imgyolo)
            # print(label ,boxes)        
            not_ship_time = not_ship_time+1   
            not_targe_time= not_targe_time+1
            # if len(label) != 0 and (not list_equal([1],label)) and (not list_equal([1,2],label)) and (not list_equal([2],label)) :
            if len(label) != 0:
                print("====== Ship in Scream ======")
                not_ship_time=0
                not_targe_time=0  
                if  1 in label: 
                    trun_r=0
                    not_targe_time=0
                    not_ship_time=0
                    targe_flage=1
                    ship_flag=2
                    print("====== target ======")
                    dx, dy=avg_get(boxes[len(label)-1][0]*1600+x1,boxes[len(label)-1][1]*800+y1,boxes[len(label)-1][2]*1600+x1,boxes[len(label)-1][3]*800+y1)   
                    # print(" box ",boxes[1][0]*1600+200+x1,boxes[1][1]*800+100+y1,boxes[1][2]*1600+200+x1,boxes[1][3]*800+100+y1)
                    # print(" dx dy ",dx, dy)
                    # print(" avgx avgy ",avgx, avgy)
                    ix, iy=pixel_to_int(dx-avgx,dy-avgy)
                    mouse_move(ix, iy)
                    time.sleep(0.5)
                    pyautogui.mouseDown(button='left')
                    pyautogui.mouseUp(button='left')
                    print(" Fire !!!!!!!",ix, iy)
                    fire_times=fire_times+1
                    testfire = 0
                    # time.sleep(0.25)
                else:
                    direct.press('p')
                    fire_times=0
                    time.sleep(1.5)
                    testfire = testfire+1
                    print("testfire",testfire)
                    if testfire > 30:
                        testfire = 0
                        direct.press('l')       
                        time.sleep(0.5)  
                        direct.press('c') 

                if fire_times > 80:
                    print("====== change to another Ship ======")
                    fire_times=0
                    if len(label)>2:
                        dsx, dsy=avg_get(boxes[1][0]*1600+x1,boxes[1][1]*800+y1,boxes[1][2]*1600+x1,boxes[1][3]*800+y1)   
                        isx, isy=pixel_to_int(dsx-avgx,dsy-avgy)
                        mouse_move(isx, isy)
                        time.sleep(0.5)
                        direct.press('l')         
                        # pyautogui.mouseDown(button='left')
                        # pyautogui.mouseUp(button='left')
                        direct.press('c') 
                    else:
                        continue
                if fire_times>130:
                    self.serch_the_ship()

                    # time.sleep(0.5)         
                                   
            if not_ship_time>10 and not_targe_time > 10 :
                print("====== search the  ship ======")  
                # k.press('space')
                # time.sleep(0.5)        
                fire_times=0
                if trun_r <5:
                    trun_r=trun_r+1
                    offset=2000
                else :
                    offset =-2000
                mouse_move(offset,0)
                time.sleep(0.5)
                # direct.move(offset,None)
                direct.press('l')
                time.sleep(0.5)
                # k.release('space')
                direct.press('c')
            time.sleep(0.5)
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
