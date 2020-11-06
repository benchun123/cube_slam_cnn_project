
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data_path = "/home/benchun/benchun/dataset/cubes/training_10"
## training: all: 18112, frame: 0-3, sample in each frame: 4094
## training_2: all: 172, frame: 3, sample in each frame: 174
## training_3: all: 6020, frame: 3, sample in each frame: 6020
## training_4: all: 656, frame: 0-3, sample in each frame: 174
## training_5: all: 21912, frame: 0-3, sample in each frame: 6020
## training_6: all: 9097, frame: 0-57, sample in each frame: 174
## training_7: all: 1160, frame: 0-57, collect from ros: 20
## training_8: all: 1936, frame: 0-57, sample in each frame: 20
## training_9: all: 975, frame: 0-57, sample in each frame: 10
## training_10: all: 2869, frame: 0-57, sample in each frame: 30
img_path = data_path + '/image_2'
label_path = data_path + '/label_2'
IDLst = [x.split('.')[0] for x in sorted(os.listdir(label_path))]
print(len(IDLst))
buffer = []
iou_data = []
pos_data = []
for idx in range(len(IDLst)):
    # print(IDLst[idx])
    # imgID = img_path + '/%s.jpg' % IDLst[idx]
    # img = cv2.imread(imgID, cv2.IMREAD_COLOR)
    # crop = cv2.resize(src=img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    with open(label_path + '/%s.txt' % IDLst[idx], 'r') as f:
        for line in f:
            line = line[:-1].split(' ')
            for i in range(1, len(line)):
                line[i] = float(line[i])
            Class = line[0]
            Pos = [line[1], line[2], line[3]]
            Yaw = line[4]
            IoU = line[8]  # added benchun 20200927 for better loss
            top_left = (int(round(line[9])), int(round(line[10])))
            bottom_right = (int(round(line[11] + line[9])), int(round(line[12] + line[10])))
            Box_2D = [top_left, bottom_right]
            buffer.append({
                'Class': Class,
                'Box_2D': Box_2D,
                'Pos': Pos,
                'Yaw': Yaw,
                'IoU': IoU,
            })
            iou_data.append(IoU)
            pos_data.append(Pos)
# plt.plot(iou_data)
plt.hist(iou_data, bins=20, range=(0,1))
plt.savefig('train_data/data.png')
plt.show()
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter([row[0] for row in pos_data], [row[1] for row in pos_data], [row[2] for row in pos_data])
# # for x in range (10):
# #     pos_data_slice = pos_data[x*17:x*17+17]
# #     print(len(pos_data))
# #     ground_truth = [-1.52992, 0.456396, 0.281669]
# #     fig = plt.figure(x)
# #     ax = Axes3D(fig)
# #     ax.scatter([row[0] for row in pos_data_slice], [row[1] for row in pos_data_slice], [row[2] for row in pos_data_slice])
# #     for i, row in enumerate(pos_data_slice):
# #         error = [row[0]-ground_truth[0], row[1]-ground_truth[1], row[2]-ground_truth[2]]
# #         distance = math.sqrt(error[0]*error[0]+error[1]*error[1]+error[2]*error[2])
# #         print(i, distance, error)
# plt.show()