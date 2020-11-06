import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable

import numpy as np
import os
import time
import datetime
from visdom import Visdom
from random import shuffle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
viz = Visdom() # online loss: run python -m visdom.server # http://localhost:8097
viz.line([0.], [0], win='shift_loss', opts=dict(title='shift_loss')) # inital
viz.scatter(X=np.asarray([[0.0, 0.0]]), win='shift_eval', opts=dict(title='shift_eval')) # inital

class shiftnet(nn.Module):
    def __init__(self, in_dim=13, n_hidden_1=1024, n_hidden_2=1024, out_dim=3):
        super(shiftnet, self).__init__()
        self.linear_1 = nn.Linear(in_dim, n_hidden_1)
        self.linear_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.linear_3 = nn.Linear(n_hidden_2, out_dim)
        # self.linear_1 =nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True)) # add relu
        # self.linear_1 =nn.sequential(nn.linear(in_dim, n.hidden_1), nn.BtachNormld(n_hidden_1), nn.Relu(True)) # add relu and norm

    def forward(self, x):
        out = self.linear_1(x)
        out = self.linear_2(out)
        out = self.linear_3(out)
        return out

# def test():
#     net = shiftnet()
#     summary(net, (1,10))
#     # y = net(torch.randn(1, 3, 224, 224))
#     # X = torch.rand((1, 3, 224, 224))
#     # for name, layer in net.named_children():
#     #     X = layer(X)
#     #     print(name, ' output shape:\t', X.shape)
# test()


class BatchDataset:
    def __init__(self, path, batchSize = 1 , mode='train'):
        self.label_path = path + '/label_2'
        self.IDLst = [x.split('.')[0] for x in sorted(os.listdir(self.label_path))]
        self.batchSize = batchSize
        self.mode = mode
        if mode == "train" :
            self.idx = 0
            self.num_of_patch = 172 # 166 #6560
        else:
            self.idx = 0
            self.num_of_patch = 6020 #4000 #6560

    def getTraindata(self):
        batch = np.zeros([self.batchSize, 13], np.float)
        trans = np.zeros([self.batchSize, 3], np.float)
        buffer = []
        for one in range(self.batchSize):
            with open(self.label_path + '/%s.txt'%self.IDLst[self.idx], 'r') as f:
                for line in f:
                    line = line[:-1].split(' ')
                    for i in range(1, len(line)):
                        line[i] = float(line[i])
                    Class = line[0]
                    Location = [line[1], line[2], line[3]]  # x, y, z
                    Ry = (line[4]) / np.pi * 180  # object yaw
                    Dimension = [line[5], line[6], line[7]]  # height, width, length
                    IoU = line[8]
                    top_left = (int(round(line[9])), int(round(line[10])))
                    bottom_right = (int(round(line[11]+line[9])), int(round(line[12]+line[10])))
                    Box_2D = [top_left, bottom_right]
                    buffer.append({
                            'Class': Class,
                            'Box_2D': Box_2D,
                            'IoU': IoU,
                            'Location': Location,
                            # 'Ry_sin': Ry_sin,
                            # 'Dimension': Dimension,
                            # 'transtoworld': transtoworld,
                    })

            batch[one, :] = line
            buff_data = buffer[one]
            trans[one, :] = buff_data['Location']

            if self.idx + 1 < self.num_of_patch:
                self.idx += 1
            else:
                self.idx = 0

        return batch, trans

def VDL_loss(predict, groundtruth, batch_data): # only for batch=1,
    # predict = [xyz], shape = [batch, 3]
    # groundtruth = [xyz], shape = [1, 3]
    # data_exclude_trans = [xyz, yaw, lwh], shape = [batch, 7]
    batch_data_size, _ = batch_data.size()
    data_exclude_trans = batch_data[:, 4:8]# predict[4:7] and groundtruth[4:7] should be the same
    loss = torch.ones(batch_data_size)
    for one in range(batch_data_size):
        yaw, l, w, h = data_exclude_trans[one] # predict[4:7] and groundtruth[4:7] should be the same
        print("yaw: ", yaw)
        cos_yaw = torch.cos(yaw) # why use torch.cos?
        sin_yaw = torch.sin(yaw)
        rotation_matrix = torch.tensor([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
        print("rotation_matrix: ", rotation_matrix)
        trans_delta = torch.sub(groundtruth[one, 0:3], predict[one, 0:3])
        print("trans_delta: ", trans_delta)
        trans_global = torch.matmul(rotation_matrix, trans_delta)
        print("trans_global: ", trans_global)
        areaxyz = torch.tensor([w*h, w*l, h*l])
        print("areaxyz: ", areaxyz)
        loss[one] = torch.matmul(areaxyz, torch.abs(trans_global))
        print("loss:", loss)
    # loss = w*h*abs(trans_global(0)) + w*l*abs(trans_global(1)) + h*l*abs(trans_global(2));
    loss = torch.sum(loss)/batch_data_size
    return loss

# x = torch.tensor([[1., 2., 3.], [2,3,4]])
# y = torch.tensor([[10, 10, 10], [10, 10, 10]])
# other_data = torch.tensor([[10, 10, 10, 0., 0., 6., 7., 8., 9., 10], [10, 10, 10, 0., 0., 6., 7., 8., 9., 10]])
# k = VDL_loss(x, y, other_data)
# print("loss:", k)


if __name__ == '__main__':
    data_path = "/home/benchun/benchun/dataset/cubes"
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode = 'train'
    net = shiftnet()
    if mode == "train":
        net.train()
        # if len(model_lst) == 0:
        #     print ('No previous model found, start training')
        #     net.train()
        # else:
        #     print ('Find previous model %s'%model_lst[-1])
        #     params = torch.load(store_path + '/%s'%model_lst[-1])
        #     net.load_state_dict(params)
        #     net.train()

    loss_stream_file = open('train_data/Loss_record_shiftnet.txt', 'w')
    lr, num_epochs, batch_size = 0.0001, 15, 2
    data = BatchDataset(data_path + '/training_2', batch_size, mode='train')

    net = net.to(device)
    print("training on ", device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    iter_each_time = round(float(data.num_of_patch) / batch_size)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for i in range(int(iter_each_time)):
            batch, transGT = data.getTraindata() #numpy ndarray
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).to(device) #tensor
            transGT = Variable(torch.FloatTensor(transGT), requires_grad=False).to(device)

            trans = net(batch)
            trans_loss = VDL_loss(trans, transGT, batch)

            optimizer.zero_grad()
            trans_loss.backward()
            optimizer.step()

            train_l_sum += trans_loss.cpu().item()
            # train_acc_sum += (iou.argmax(dim=1) == iouGT).sum().cpu().item()
            n += transGT.shape[0]
            print("transGT.shape[0]:", transGT.shape[0])
            batch_count += 1
            viz.line([trans_loss.cpu().item()], [epoch * iter_each_time + i], win='train_loss', update='append')

            if i % 2 == 0:
                now = datetime.datetime.now()
                now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
                # print (' %s Epoch %.2d, IoU Losss: %lf'%(now_s, i, iou_loss))
                trans = trans.cpu().data.numpy()[0, :]
                transGT = transGT.cpu().data.numpy()[0, :]
#                loss_stream_file.write('%lf %lf %lf %lf \n ' % (i, iou_loss, iou, iouGT))
        # test_acc = evaluate_accuracy(test_iter, net)
        test_acc = 0
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        # now = datetime.datetime.now()
        # now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
        # name = store_path + '/model_%s.pkl' % now_s
        # torch.save(net.state_dict(), name)
    loss_stream_file.close()