#score network framework
we provide three simple framework for score function
Resnet18: build a resnet18 and train the network
shiftnet: build a fc network and train it (not test)
loss_plot: show the training loss and evaluation result
data_distribution: check the distribution of self dataset

##environment
version: 
python: 3.5
torch: 1.4.0+cpu
visdom

## set up
virtual envrionment
use anaconda or virtual environment
I prefer virtual environment

make a torch.txt file and every time you want to use python, just source this file
(in case of comflict between python3 and ros)

the source file look as
source ~/Software/virtualenv/torch/bin/activate
export PYTHONPATH="/home/benchun/Software/virtualenv/torch/lib/python3.5/site-packages":$PYTHONPATH


