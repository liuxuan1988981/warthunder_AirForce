# coding:utf-8
# load own dataset
import numpy as np
from src import Log
from utils import tools
import random
import cv2
from src import Log
from os import path
from Data_augment.Color_enhancement import Color_enhancement
from Data_augment.Erase import Erase
from Data_augment.Flip import Flip
from Data_augment.Translate import Translate

class Data():
    def __init__(self, data_file, voc_names_file, class_num, batch_size, anchors, width=608, height=608, data_debug=False, is_tiny=False):
        self.data_file = data_file  # data dictionary
        self.class_num = class_num  # classify number
        self.batch_size = batch_size
        self.anchors = np.asarray(anchors).astype(np.float32).reshape([-1, 2]) / [width, height]     #[9,2]
        print("anchors:\n", self.anchors)

        self.debug_img = data_debug
        self.is_tiny = is_tiny

        self.imgs_path = []
        self.labels_path = []

        self.num_batch = 0     # total batch number
        self.num_imgs = 0       # total number of images

        self.width = width
        self.height = height

        self.names_dict = tools.word2id(voc_names_file)    # dictionary of name to id
        
        self.steps_per_epoch = 1000 
        # 初始化数据增强策略的参数
        
        # ################## data augment ##################
        self.flip = Flip()
        self.flip_img = 0.5    # probility of flip image

        self.gray_img = 0.0        # probility to gray picture
        
        self.label_smooth = 0.001    # label smooth delta  

        # random erase
        self.erase = Erase(max_erase=5, max_w=30, max_h=30)
        self.erase_img = 0.5        # probility of random erase some area

        # color enhance
        self.color_enhance = Color_enhancement()

        # rotate : not use
        self.rotate_img = 0.0   # probility to rorate the image 

        # translate
        self.translate = Translate(self.width, self.height, pad_thresh=30)
        self.trans_img = 0.5    # probility of translate the image

        self.gasuss = 0.0       # gasuss norse
        self.__init_args()
    
    # initial all parameters
    def __init_args(self):
        Log.add_log("message: begin to initial images path")
        # init imgs path
        self.imgs_path = tools.read_file(self.data_file)
        if not self.imgs_path:
            Log.add_log("error:imgs_path not exist")
            raise ValueError("no file find in :" + str(self.data_file))
        self.num_imgs = len(self.imgs_path)
        self.steps_per_epoch = int(self.num_imgs / self.batch_size)
        Log.add_log("message:there are "+str(self.num_imgs) + " pictures in all")

        # init labels path
        for img in self.imgs_path:
            label_path = img.replace("JPEGImages", "labels")
            label_path = label_path.replace(img.split('.')[-1], "txt")
            self.labels_path.append(label_path)
        Log.add_log("message: initialize images path and corresponding labels complete")
        
        return

    # read image
    def __read_img(self, img_file):
        '''
        read img_file
        return:img(RGB)
        '''
        img = tools.read_img(img_file)
        if img is None:
            return None
        return img
    
    # 解析voc的xml文件
    def parse_voc_xml(self, label_file):
        '''
        return: ids and boxes are integrate list
        '''
        # 这里的box的 [name_id, xmin, ymin, xmax, ymax] 是整数
        # contents = tools.parse_voc_xml(xml_file, names_dict=self.names_dict, get_ori=True)  
        # !! 注意这里是字符串的浮点数 [id, x,y,w,h]
        contents = tools.read_file(label_file)
        ids = []
        boxes = []
        for content in contents:
            content = content.split()
            ids.append(int(content[0]))
            boxes.append(content[1:])
        return ids, boxes

    # transfor the boxes from integer to float
    def __trans_boxes2float(self, boxes_int, ori_w, ori_h):
        '''
        return:[[x, y, w, h]] float
        '''
        trans_boxes = []
        for box in boxes_int:
            new_box = self.__trans_box2float(box, ori_w, ori_h)
            trans_boxes.append(new_box)
        return trans_boxes
    
    # transfor the boxes from float to integer
    def __trans_boxes2int(self, boxes_float, ori_w, ori_h):
        '''
        boxes_float:[
            [x,y,w,h],
            []
        ]
        return:[[xmin, ymin, xmax, ymax]] float
        '''
        trans_boxes = []
        for box in boxes_float:
            new_box = self.__trans_box2int(box, ori_w, ori_h)
            trans_boxes.append(new_box)
        return trans_boxes

    # transfor one box from integer to float
    def __trans_box2float(self, box_int, width, height):
        '''
        return:[x, y, w, h] float
        '''
        xmin = box_int[0]
        ymin = box_int[1]
        xmax = box_int[2]
        ymax = box_int[3]

        x = (xmax + xmin) / 2.0 / width
        w = (xmax - xmin) / width
        y = (ymax + ymin) / 2.0 / height
        h = (ymax - ymin) / height
        
        return [x, y, w, h]

    # transfor one box from float to integer
    def __trans_box2int(self, box_float, width, height):
        '''
        box_float:[x,y,w,h],
        return:[xmin,ymin, xmax, ymax] int
        '''
        x = float(box_float[0].strip()) * width
        y = float(box_float[1].strip()) * height
        w = float(box_float[2].strip()) * width
        h = float(box_float[3].strip()) * height
        
        xmin = x-w/2
        ymin = y-h/2
        xmax = x+w/2
        ymax = y+h/2
        
        return [xmin, ymin, xmax, ymax]
    
    # remove the image and xml file which have no targets
    def __remove(self, img_file, xml_file):
        self.imgs_path.remove(img_file)
        self.labels_path.remove(xml_file)
        self.num_imgs -= 1
        if not len(self.imgs_path) == len(self.labels_path):
            Log.add_log("error:after remove the file:{} , the number of label and image is not equal".format(img_file))
            assert(0)
        return 

    # make the label of one image
    def __make_label(self, ids, boxes, anchors, class_num):
        '''
        ids:the id of classify to the box
        boxes:[[x,y,w,h]]
        anchors:the anchors
        class_num:
        return:label_y1, label_y2
        '''
        label_y13 = np.zeros((self.height // 32, self.width // 32, 3, 5+class_num), np.float32)
        label_y26 = np.zeros((self.height // 16, self.width // 16, 3, 5+class_num), np.float32)
        label_y52 = np.zeros((self.height // 8, self.width // 8, 3, 5+class_num), np.float32)

        if self.label_smooth:
            label_y13[:, :, :, 4] = self.label_smooth  / (self.class_num + 1)
            label_y26[:, :, :, 4] = self.label_smooth  / (self.class_num + 1)
            label_y52[:, :, :, 4] = self.label_smooth / (self.class_num + 1)
        
        if self.is_tiny:
            y_true = [label_y26, label_y13]
            ratio = {0:16, 1:32}
        else:
            y_true = [label_y52, label_y26, label_y13]
            ratio = {0:8, 1:16, 2:32}

        for i in range(len(ids)):
            label_id = int(ids[i])
            box = np.asarray(boxes[i]).astype(np.float32)  

            best_giou = 0
            best_index = 0
            for i in range(len(anchors)):
                min_wh = np.minimum(box[2:4], anchors[i])
                max_wh = np.maximum(box[2:4], anchors[i])
                giou = (min_wh[0] * min_wh[1]) / (max_wh[0] * max_wh[1])
                if giou > best_giou:
                    best_giou = giou
                    best_index = i
            
            # 012->0, 345->1
            x = int(np.floor(box[0] * self.width / ratio[best_index // 3]))
            y = int(np.floor(box[1] * self.height / ratio[best_index // 3]))
            k = best_index % 3

            y_true[best_index // 3][y, x, k, 0:4] = box
            delta = self.label_smooth
            label_value = 1.0  if not self.label_smooth else ((1-delta) + delta / (self.class_num+1))
            y_true[best_index // 3][y, x, k, 4:5] = label_value
            y_true[best_index // 3][y, x, k, 5+label_id] = label_value
        
        return label_y13, label_y26, label_y52

    # data augment
    def __data_augment(self, img, boxes, ids):
        height, width, _ = img.shape

        # flip
        if np.random.random() < self.flip_img:
            img, boxes = self.flip.run(img, boxes)
            pass

        # translate
        if np.random.random() < self.trans_img:
            x_shift = np.random.randint(-2, 3) * 30
            y_shift = np.random.randint(-2, 3) * 30
            img, boxes, ids = self.translate.run(img, boxes, x_shift, y_shift, ids=ids, max_width=width, max_height=height)
            pass

        # random erase
        if np.random.random() < self.erase_img:
            img = self.erase.run(img)
            pass
        
        # gray
        if np.random.random() < self.gray_img:
            tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # to 3 channel
            img = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)

        # color enhance
        color = np.random.randint(8, 13) / 10.0 if np.random.random() > 0.5 else 1.0
        contrast = np.random.randint(8, 16) / 10.0 if np.random.random() > 0.5 else 1.0            
        img = self.color_enhance.run(img, random_enhance=False, color=color, contrast=contrast)
        
        # guass noise : not use
        if np.random.random() < self.gasuss:
            noise = np.random.normal(0, 0.01 * 255, img.shape)
            img = img+noise
            img = np.clip(img, 0, 255)
            img = np.uint8(img)
        return img, boxes, ids

    # load the data of one image
    def __load_one_data(self, img_name, label_name):         
        '''
        return: img, [label_y1, label_y2]
        '''   
        # read image
        img = self.__read_img(img_name)      
        if img is None:
            Log.log("warning:VOC image'" + img_name + "'is not a file")
            self.__remove(img_name, label_name)
            return None, None

        # read xml file      
        # ! 注意这里是浮点数,字符串浮点数 [x,y,w,h]
        ids, boxes = self.parse_voc_xml(label_name)
        if len(boxes) is 0:
            Log.log("warning:VOC file'" + label_name + "'have no targets")
            self.__remove(img_name, label_name)
            return None, None

        height, width, _ = img.shape
        
        # 转为int [xmin,ymin, xmax, yman]
        boxes = self.__trans_boxes2int(boxes, width, height)

        # data augment
        img, boxes, ids = self.__data_augment(img, boxes, ids)
        test_img = img
    
        # boxes to float
        boxes = self.__trans_boxes2float(boxes, width, height)

        # resize
        if height != self.height or width != self.width:
            img = cv2.resize(img, (self.width, self.height))

        # image to float
        img = img.astype(np.float32)
        img = img/255.0
        
        # make label
        label_y1, label_y2, label_y3 = self.__make_label(ids, boxes, self.anchors, self.class_num)

        if label_y1 is None:
            Log.log("warning:VOC file'" + label_name + "'is not a file")
            return None, None

        # show the image
        if self.debug_img:
            test_img = tools.draw_box_float(test_img, boxes)
            cv2.imshow("test_img", test_img)
            cv2.waitKey(0)
        return img, [label_y1, label_y2, label_y3]

    def get_data(self):
        return self.__get_data()

    # the function just for tf.data.dataset 
    def load_tf_batch_data(self, tf_inputs):
        '''
            inp=[(imgs_batch, xmls_batch)],
            Tout=[tf.float32, tf.float32, tf.float32]),
        '''
        # use tf.data to load the image and label
        imgs_batch, xmls_batch = tf_inputs
        decode_type = 'UTF-8'
        imgs = []
        labels_y13, labels_y26, labels_y52 = [], [], []

        imgs_batch = imgs_batch.tolist()
        for i in range(len(imgs_batch)):
            img_name = imgs_batch[i]
            img_name = img_name.decode(decode_type)     # chinese also working
            label_name = xmls_batch[i]
            label_name = label_name.decode(decode_type)
            
            # load one data
            img, labels = self.__load_one_data(img_name, label_name)
            if img is None:
                assert(0)

            label_y1, label_y2, label_y3 = labels

            imgs.append(img)
            labels_y13.append(label_y1)
            labels_y26.append(label_y2)
            labels_y52.append(label_y3)

        imgs = np.asarray(imgs)
        labels_y13 = np.asarray(labels_y13)
        labels_y26 = np.asarray(labels_y26)

        # if is yolo_tiny
        if not self.is_tiny:
            labels_y52 = np.asarray(labels_y52)
            return imgs, labels_y13, labels_y26, labels_y52
        return imgs, labels_y13, labels_y26

    # load batch_size data
    def __get_data(self):        
        imgs = []
        labels_y13, labels_y26, labels_y52 = [], [], []

        count = 0
        while count < self.batch_size:
            curr_index = random.randint(0, self.num_imgs - 1)
            img_name = self.imgs_path[curr_index]
            label_name = self.labels_path[curr_index]
            
            # read one data
            img, labels = self.__load_one_data(img_name, label_name)
            if img is None:
                continue
            label_y1, label_y2, label_y3 = labels

            imgs.append(img)
            labels_y13.append(label_y1)
            labels_y26.append(label_y2)
            labels_y52.append(label_y3)
            count += 1

        self.num_batch += 1
        imgs = np.asarray(imgs)
        labels_y13 = np.asarray(labels_y13)
        labels_y26 = np.asarray(labels_y26)
        
        if not self.is_tiny:
            labels_y52 = np.asarray(labels_y52)
            return imgs, labels_y13, labels_y26, labels_y52
        return imgs, labels_y13, labels_y26
        

    # iterator
    def __next__(self):
        '''
        迭代获得一个 batch 的数据
        '''
        return self.__get_data()

    def __len__(self):
        return self.num_imgs

    # data generater for keras
    def data_generater(self):
        while True:
            if self.is_tiny:
                imgs, label_13, label_26 = self.__get_data()
                yield (imgs, [label_13, label_26])
            else:
                imgs, label13, label_26, label52 = self.__get_data()
                yield (imgs, [label13, label_26, label52])
