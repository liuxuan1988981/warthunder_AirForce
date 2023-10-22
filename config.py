# coding:utf-8
# configuration file

# ############# Basic configuration. #############
# width = 1024
# height = 512                     # image size
width = 832
height = 416                     # image size
batch_size = 8
batch_size_tiny = 32
total_epoch = 30000      # total epoch
save_per_epoch = 50        # per save_step save one model
data_debug = False       # load data in debug model (show pictures when loading images)
cls_normalizer = 1.0    # Loss coefficient of confidence
iou_normalizer = 0.07   # loss coefficient of ciou
iou_thresh = 0.5     # 
prob_thresh = 0.25      # 
score_thresh = 0.25     # 
val_score_thresh = 0.5      # 
val_iou_thresh = 0.213            # 
max_box = 50                # 
save_img = False             # save the result image when test the net

# ############# log #############
log_dir = './log'
log_name = 'log.txt'
loss_name = 'loss.txt'

# configure the leanring rate
lr_init = 2e-4/10                      # initial learning rate	# 0.00261
lr_lower =1e-6/10                  # minimum learning rate    
lr_type = 'piecewise'   # type of learning rate( 'exponential', 'piecewise', 'constant')
piecewise_boundaries = [600, 6000, 8000]   #  for piecewise
piecewise_values = [2e-4/10, 0.00032/10, 2e-4/10, 1e-4/10]   # piecewise learning rate

# configure the optimizer
optimizer_type = 'adam' # type of optimizer
momentum = 0.949          # 
weight_decay = 0

# ############## training on own dataset ##############
class_num = 1
# anchors = 16,15, 17,16, 34,31, 39,45, 49,46, 60,51, 69,53, 80,59, 88,72
# anchors =15,16, 16,15, 35,47, 55,48, 45,60, 72,56, 83,64, 83,74, 86,87
anchors =9,11, 10,10, 10,11, 10,12, 10,12, 12,11, 11,12, 12,13, 13,15
voc_class_num = 1
voc_anchors = 9,11, 10,10, 10,11, 10,12, 10,12, 12,11, 11,12, 12,13, 13,15
voc_names = "./data/train.names"                             # the names of voc dataset

# ############## train on VOC ##############
# voc_root_dir = ["/media/random/数据/Dataset/VOC/VOC2007",
#                             "/media/random/数据/Dataset/VOC/VOC2012"]  # root directory of voc dataset
# voc_class_num = 20 
# voc_anchors = 10,13,  16,30, 33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
voc_test_dir = "./voc_test_pic"                                                 # test pictures directory for VOC dataset
voc_save_dir = "./voc_save"                                                     # the folder to save result image for VOC dataset
voc_model_path = "./VOC"                                                        # the folder to save model for VOC dataset
voc_model_name = "voc"                                          # the model name for VOC dataset
# voc_names = "./data/voc.names"                             # the names of voc dataset
