import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import sys

from torch.autograd import Variable
import datetime
import os
import numpy as np
import cv2
from visdom import Visdom
from random import shuffle

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
viz = Visdom() # online loss: run python -m visdom.server # http://localhost:8097
viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss')) # inital
viz.scatter(X=np.asarray([[0.0, 0.0]]), win='eval', opts=dict(title='eval_error')) # inital

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
# net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
# net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 1)))
net.add_module("iou", nn.Sequential(FlattenLayer(),
                                    nn.Linear(512 * 7 * 7, 256),
                                    nn.ReLU(True),
                                    nn.Dropout(),
                                    nn.Linear(256, 256),
                                    nn.ReLU(True),
                                    nn.Dropout(),
                                    nn.Linear(256, 1)
                                    ))


def ResNet():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    model.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    model.add_module("resnet_block2", resnet_block(64, 128, 2))
    model.add_module("resnet_block3", resnet_block(128, 256, 2))
    model.add_module("resnet_block4", resnet_block(256, 512, 2))

    # model.add_module("global_avg_pool", GlobalAvgPool2d())
    # model.add_module("fc", torch.nn.Sequential(FlattenLayer(), torch.nn.Linear(512, 1)))

    model.add_module("linear1", nn.Linear(512 * 7 * 7, 256))
    model.add_module("relu1", nn.ReLU(True))
    model.add_module("dropout1", nn.Dropout())
    model.add_module("linear2", nn.Linear(256, 256))
    model.add_module("relu2", nn.ReLU(True))
    model.add_module("dropout2", nn.Dropout())
    model.add_module("linear3", nn.Linear(256, 1))

# X = torch.rand((1, 3, 224, 224))
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, ' output shape:\t', X.shape)
# summary(net, (3, 224, 224))
# print(net)

class BatchDataset:
    def __init__(self, path, batchSize = 1 , mode='train'):
        self.img_path = path + '/image_2'
        self.label_path = path + '/label_2'
        self.IDLst = [x.split('.')[0] for x in sorted(os.listdir(self.img_path))]
        shuffle(self.IDLst) # added 20201102 random shuffle
        self.batchSize = batchSize
        self.mode = mode
        self.imgID = None
        #self.info = self.getBatchInfo()
        #self.Total = len(self.info)
        if mode == 'train':
            self.idx = 0
            self.num_of_patch = 2869 # 166 #6560
        else:
            self.idx = 0
            self.num_of_patch = 9097 #656 #4000 #6560

    def getTraindata(self):
        batch = np.zeros([self.batchSize, 3, 224, 224], np.float)
        iou = np.zeros([self.batchSize, 1], np.float)
        buffer = []
        for one in range(self.batchSize):
            with open(self.label_path + '/%s.txt'%self.IDLst[self.idx], 'r') as f:
                for line in f:
                    line = line[:-1].split(' ')
                    for i in range(1, len(line)):
                        line[i] = float(line[i])
                    Class = line[0]
                    IoU = line[8]  # added benchun 20200927 for better loss
                    top_left = (int(round(line[9])), int(round(line[10])))
                    bottom_right = (int(round(line[11]+line[9])), int(round(line[12]+line[10])))
                    Box_2D = [top_left, bottom_right]
                    buffer.append({
                            'Class': Class,
                            'Box_2D': Box_2D,
                            'IoU': IoU,
                        })

            buff_data = buffer[one]
            imgID = self.img_path + '/%s.jpg' % self.IDLst[self.idx]
            if imgID != None:
                img = cv2.imread(imgID, cv2.IMREAD_COLOR).astype(np.float) / 255
                img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
                img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
                img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # cv2.namedWindow('GG')
                # cv2.imshow('origin', img)
                # cv2.waitKey(0)

            pt1 = buff_data['Box_2D'][0]
            pt2 = buff_data['Box_2D'][1]
            crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
            # crop = img   # added benchun 20201016, not crop
            # cv2.imshow('cut', crop)
            # cv2.waitKey(0)
            crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            batch[one, 0, :, :] = crop[:, :, 2]
            batch[one, 1, :, :] = crop[:, :, 1]
            batch[one, 2, :, :] = crop[:, :, 0]
            iou[one, :] = buff_data['IoU']
            # cv2.imshow('resize', crop)
            # cv2.waitKey(0)
            # print("crop", crop)
            # print(batch[one, :, :, :])

            if self.idx + 1 < self.num_of_patch:
                self.idx += 1
            else:
                self.idx = 0

        return batch, iou

    # def getEvaldata(self):
    #     batch = np.zeros([1, 3, 224, 224], np.float)
    #     iou = np.zeros([1, 1], np.float)
    #     for one in range(1):
    #         with open(self.label_path + '/%s.txt'%self.IDLst[self.idx], 'r') as f:
    #             for line in f:
    #                 line = line[:-1].split(' ')
    #                 for i in range(1, len(line)):
    #                     line[i] = float(line[i])
    #                 Class = line[0]
    #                 IoU = line[8]
    #                 top_left = (int(round(line[9])), int(round(line[10])))
    #                 bottom_right = (int(round(line[11])), int(round(line[12])))
    #                 Box_2D = [top_left, bottom_right]
    #                 buffer.append({
    #                         'Class': Class,
    #                         'Box_2D': Box_2D,
    #                         'IoU': IoU,
    #                     })
    #
    #         buff_data = buffer[one]
    #         imgID = self.img_path + '/%s.jpg' % self.IDLst[self.idx]
    #         if imgID != None:
    #             img = cv2.imread(imgID, cv2.IMREAD_COLOR).astype(np.float) / 255
    #             img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
    #             img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    #             img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
    #             # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             # cv2.namedWindow('GG')
    #             # cv2.imshow('GG', img)
    #             # cv2.waitKey(0)
    #
    #         pt1 = buff_data['Box_2D'][0]
    #         pt2 = buff_data['Box_2D'][1]
    #         crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
    #         crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    #         batch[one, 0, :, :] = crop[:, :, 2]
    #         batch[one, 1, :, :] = crop[:, :, 1]
    #         batch[one, 2, :, :] = crop[:, :, 0]
    #         iou[one, :] = buff_data['IoU']
    #
    #         if self.idx + 1 < self.num_of_patch:
    #             self.idx += 1
    #         else:
    #             self.idx = 0
    #
    #     return batch, iou

if __name__ == '__main__':
    data_path = "/home/benchun/benchun/dataset/cubes"
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]

    mode = 'train'
    if mode == "train":
        if len(model_lst) == 0:
            print ('No previous model found, start training')
            net = net
            net.train()
            # model = net.Model(features=vgg.features, bins=bins).cuda()
        else:
            print ('Find previous model %s'%model_lst[-1])
            # model = Model.Model(features=vgg.features, bins=bins)
            # model = Model.Model(features=vgg.features, bins=bins).cuda()
            params = torch.load(store_path + '/%s'%model_lst[-1])
            net.load_state_dict(params)
            net.train()

        loss_stream_file = open('train_data/Loss_record.txt', 'w')
        lr, num_epochs, batch_size = 0.0001, 15, 8
        data = BatchDataset(data_path + '/training_10', batch_size, mode='train')

        net = net.to(device)
        print("training on ", device)
        # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        iter_each_time = round(float(data.num_of_patch) / batch_size)
        for epoch in range(num_epochs):
            # data = BatchDataset(data_path + '/training_8', batch_size, mode='train') #enable shuffle each image
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for i in range(int(iter_each_time)):
                batch, iouGT = data.getTraindata()
                batch = Variable(torch.FloatTensor(batch), requires_grad=False).to(device)
                iouGT = Variable(torch.FloatTensor(iouGT), requires_grad=False).to(device)
                iou = net(batch)
                # iou_LossFunc = torch.nn.MSELoss()
                iou_LossFunc = torch.nn.SmoothL1Loss()
                iou_loss = iou_LossFunc(iou, iouGT)

                optimizer.zero_grad()
                iou_loss.backward()
                optimizer.step()

                train_l_sum += iou_loss.cpu().item()
                train_acc_sum += (iou.argmax(dim=1) == iouGT).sum().cpu().item()
                n += iouGT.shape[0]
                batch_count += 1
                viz.line([iou_loss.cpu().item()], [epoch*iter_each_time+i],win='train_loss', update='append')

                if i % 2 == 0:
                    now = datetime.datetime.now()
                    now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
                    # print (' %s Epoch %.2d, IoU Loss: %lf'%(now_s, i, iou_loss))
                    iou = iou.cpu().data.numpy()[0, :]
                    iouGT = iouGT.cpu().data.numpy()[0, :]
                    loss_stream_file.write('%lf %lf %lf %lf \n ' %(i, iou_loss, iou, iouGT))
            # test_acc = evaluate_accuracy(test_iter, net)
            test_acc = 0
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            now = datetime.datetime.now()
            now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
            name = store_path + '/model_%s.pkl' % now_s
            torch.save(net.state_dict(), name)

            # An example input you would normally provide to your model's forward() method.
            example = torch.rand(1, 3, 224, 224)
            # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
            net.eval()
            traced_script_module = torch.jit.trace(net, example)
            traced_script_module.save("traced_resnet_model.pt")
            net.train()
        loss_stream_file.close()

    # else:
        model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]
        if len(model_lst) == 0:
            print ('No previous model found, please check it')
            exit()
        else:
            print ('Find previous model %s'%model_lst[-1])
            params = torch.load(store_path + '/%s'%model_lst[-1])
            net.load_state_dict(params)
            net.eval()

        with torch.no_grad():
            iou_stream_file = open('train_data/DNN_result.txt', 'w')
            iou_error = []
            data = BatchDataset(data_path + '/training_6', batchSize=1, mode='eval')
            for i in range(data.num_of_patch):
                batch, iouGT = data.getTraindata()
                batch = Variable(torch.FloatTensor(batch), requires_grad=False)
                # batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()

                iou = net(batch)
                iou = iou.cpu().data.numpy()
                iou_err = np.mean((np.array(iouGT) - iou))
                # iou_err = np.mean(abs(np.array(iouGT) - iou))
                iou_error.append(iou_err)

                viz.scatter(X=np.array([[iouGT[0,0],iou[0,0]]]), win='eval', update='append')
                # print ('frame: %lf Iou error: %lf %lf %lf '%(i, iou_err, iou, iouGT))
                iou_stream_file.write('%lf %lf %lf %lf\n '%(i,iou_err,  iou, iouGT))

                if i % 1000 == 0:
                    now = datetime.datetime.now()
                    now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
                    print('------- %s %.5d -------' % (now_s, i))
                    print('IoU error: %lf' % (np.mean(iou_error)))
                    print('-----------------------------')

        iou_stream_file.close()





