# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time

# Check if GPU is available
use_gpu = torch.cuda.is_available()
print('use gpu:',use_gpu)

import torch
from torch.autograd import Variable
from torchvision import transforms

import cv2
import random
import numpy as np

import torch
import torch.utils.data  as data
import os
import cv2

import torch
import torch.utils.data  as data
import os
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import tqdm
from PIL import Image
import numpy as np

import numpy as np
import imgaug.augmenters as iaa

###############################################################################################################

class ColorHintTransform(object):
  def __init__(self, size=256, mode="training"):
    super(ColorHintTransform, self).__init__()
    self.size = size
    self.mode = mode
    print("mode: ",self.mode)
    if mode == "test" or "training" or "validation":
      self.transform = transforms.Compose([transforms.ToTensor()])
    elif mode == "transform":
      self.transform = transforms.Compose([
                                           #transforms.ToPILImage(),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomAffine(30),
                                           transforms.RandomPerspective(),
                                           transforms.ColorJitter(brightness=(0.2, 2), 
                                                                   contrast=(0.3, 2), 
                                                                   saturation=(0.2, 2), 
                                                                   hue=(-0.3, 0.3)),
                                           #transforms.ColorJitter(saturation=(0.2, 3)),
                                           #transforms.ColorJitter(saturation=(0.2, 3)),
                                           #transforms.ColorJitter(brightness=(0.2, 3)),
                                           transforms.ToTensor()             
      ])
    elif mode == "transform2":
      self.transform2 = transforms.Compose([
                                           transforms.ToPILImage(),
                                           #transforms.RandomHorizontalFlip(),
                                           #transforms.RandomVerticalFlip(),
                                           #transforms.RandomAffine(30),
                                           #transforms.RandomPerspective(),
                                           transforms.ColorJitter(brightness=(0.2, 2), 
                                                                   contrast=(0.3, 2), 
                                                                   saturation=(0.2, 2), 
                                                                   hue=(-0.3, 0.3)),
                                           transforms.ToTensor()             
      ])


  def bgr_to_lab(self, img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, ab = lab[:, :, 0], lab[:, :, 1:]
    return l, ab

  def hint_mask(self, bgr, threshold=[0.95, 0.97, 0.99]):
    h, w, c = bgr.shape
    mask_threshold = random.choice(threshold)
    mask = np.random.random([h, w, 1]) > mask_threshold
    return mask

  def img_to_mask(self, mask_img):
    mask = mask_img[:, :, 0, np.newaxis] >= 255
    return mask

  def __call__(self, img, mask_img=None):
    threshold = [0.95, 0.97, 0.99]
    if (self.mode == "training") | (self.mode == "validation") | (self.mode=="transform")| (self.mode=="transform2"):
      image = cv2.resize(img, (self.size, self.size))
      
      #if self.mode=="transform":
        #plt.imshow(image)
        #plt.show()
        
      mask = self.hint_mask(image, threshold)

      hint_image = image * mask

      l, ab = self.bgr_to_lab(image)
      l_hint, ab_hint = self.bgr_to_lab(hint_image)

      return self.transform(l), self.transform(ab), self.transform(ab_hint)

    elif self.mode == "testing":
      image = cv2.resize(img, (self.size, self.size))
      hint_image = image * self.img_to_mask(mask_img)

      l, _ = self.bgr_to_lab(image)
      _, ab_hint = self.bgr_to_lab(hint_image)

      return self.transform(l), self.transform(ab_hint)

    else:
      return NotImplementedError
      
###############################################################################################################

class ColorHintDataset(data.Dataset):
  def __init__(self, root_path, size):
    super(ColorHintDataset, self).__init__()

    self.root_path = root_path
    self.size = size
    self.transforms = None
    self.examples = None
    self.hint = None
    self.mask = None

  def set_mode(self, mode):
    self.mode = mode
    self.transforms = ColorHintTransform(self.size, mode)
    
    #print(len(os.listdir('images1/train/class'))) #os.listdir 함수는 디렉토리 안에 있는 모든 파일 이름을 리턴
    #print(len(os.listdir('images1/val/class')))
    if mode == "training":
      train_dir = os.path.join(self.root_path, "train/class")
      self.examples = [os.path.join(self.root_path, "train/class", dirs) for dirs in os.listdir(train_dir)]
    elif mode == "validation":
      val_dir = os.path.join(self.root_path, "val/class")
      self.examples = [os.path.join(self.root_path, "val/class", dirs) for dirs in os.listdir(val_dir)]
    elif mode == "testing":
      hint_dir = os.path.join(self.root_path, "hint")
      mask_dir = os.path.join(self.root_path, "mask")
      self.hint = [os.path.join(self.root_path, "hint", dirs) for dirs in os.listdir(hint_dir)]
      self.mask = [os.path.join(self.root_path, "mask", dirs) for dirs in os.listdir(mask_dir)]
    elif mode == "transform":
      train_dir = os.path.join(self.root_path, "train/class")
      self.examples = [os.path.join(self.root_path, "train/class", dirs) for dirs in os.listdir(train_dir)]   
    elif mode == "transform2":
      train_dir = os.path.join(self.root_path, "train/class")
      self.examples = [os.path.join(self.root_path, "train/class", dirs) for dirs in os.listdir(train_dir)]
    else:
      raise NotImplementedError

  def __len__(self):
    if self.mode != "testing":
      return len(self.examples)
    else:
      return len(self.hint)
      
  def __getitem__(self, idx):
    if self.mode == "testing":
      hint_file_name = self.hint[idx]
      mask_file_name = self.mask[idx]
      hint_img = cv2.imread(hint_file_name)
      mask_img = cv2.imread(mask_file_name)

      input_l, input_hint = self.transforms(hint_img, mask_img)
      sample = {"l": input_l, "hint": input_hint,
                "file_name": "image_%06d.png" % int(os.path.basename(hint_file_name).split('.')[0])}
    else:
      file_name = self.examples[idx]
      img = cv2.imread(file_name)
      l, ab, hint = self.transforms(img)
      sample = {"l": l, "ab": ab, "hint": hint}

    return sample
      
###############################################################################################################

def tensor2im(input_image, imtype=np.uint8):
  if isinstance(input_image, torch.Tensor):
      image_tensor = input_image.data
  else:
      return input_image
  image_numpy = image_tensor[0].cpu().float().numpy()
  if image_numpy.shape[0] == 1:
      image_numpy = np.tile(image_numpy, (3, 1, 1))
  image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
  return image_numpy.astype(imtype)

###############################################################################################################
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

# Change to your data root directory
root_path = "images1"
# Depend on runtime setting
use_cuda = True

train_dataset = ColorHintDataset(root_path, 128)
train_dataset.set_mode("training")
print('train_dataset length: ',len(train_dataset))

tr_train_dataset = ColorHintDataset(root_path, 128)
tr_train_dataset.set_mode("transform")
print('tr_train_dataset length: ',len(tr_train_dataset))

#tr2_train_dataset = ColorHintDataset(root_path, 128)
#tr2_train_dataset.set_mode("transform2")
#print('tr2_train_dataset length: ',len(tr2_train_dataset))

train_dataset=ConcatDataset([train_dataset, tr_train_dataset])
#train_dataset=ConcatDataset([train_dataset, tr2_train_dataset])
train_dataloader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
print('train_dataset length: ',len(train_dataset))

val_dataset = ColorHintDataset(root_path, 128)
val_dataset.set_mode("validation")
val_dataloader = data.DataLoader(val_dataset, batch_size=16, shuffle=True)
print('val_dataset length: ',len(val_dataset))

test_dataset = ColorHintDataset(root_path, 128)
test_dataset.set_mode("testing")
test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

###############################################################################################################

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel, filters=[32,64,128,256 ]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 2, 1, 1),
            nn.ReLU(),
        )
        

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output

################################################################################################################

model = ResUnet(3)
PATH="checkpoints/model-epoch-123-losses-0.000318.pth"
model.load_state_dict(torch.load(PATH))

criterion = nn.MSELoss() #Loss Function

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.7,verbose=True)

###############################################################################################################

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    
###############################################################################################################

def to_rgb_result(grayscale_input, ab_input, save_path=None, save_name=None):
  plt.clf() # clear matplotlib 
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None: 
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['result'],save_name))
    
###############################################################################################################

def validate(val_loader, model, criterion, save_images, epoch):
  model.eval()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  for i, data in enumerate(val_dataloader):
  
    if use_cuda:
      input_gray = data["l"].to('cuda')
      input_ab = data["ab"].to('cuda')
      hint = data["hint"].to('cuda')
    
    gt_image = torch.cat((input_gray, input_ab), dim=1)
    hint_image = torch.cat((input_gray, hint), dim=1)

    # Run model and record loss
    output_ab = model(hint_image) # throw away class predictions
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), hint_image.size(0))  

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 10 == 0:
      print('Validate: [{0}/{1}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(val_loader), loss=losses))

  print('Finished validation.')
  return losses.avg
  
###############################################################################################################

def train(train_loader, model, criterion, optimizer, epoch):
  print('Starting training epoch {}'.format(epoch))
  model.train()
  
  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  for i, data in enumerate(train_dataloader):
  
    if use_cuda:
      input_gray = data["l"].to('cuda')
      input_ab = data["ab"].to('cuda')
      hint = data["hint"].to('cuda')
  
    gt_image = torch.cat((input_gray, input_ab), dim=1)
    hint_image = torch.cat((input_gray, hint), dim=1)

    # Run forward pass
    output_ab = model(hint_image) 
    #copiedOutput=output_ab.data.cpu().numpy()
    loss = criterion(output_ab, input_ab) 
    losses.update(loss.item(), hint_image.size(0)) 

    # Compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #scheduler.step()

    # Print model accuracy -- in the code below, val refers to value, not validation
    if i % 25 == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i, len(train_loader), loss=losses)) 

  print('Finished training epoch {}'.format(epoch))

###############################################################################################################

# Move model and loss function to GPU
if use_gpu: 
  criterion = criterion.cuda()
  model = model.cuda()

# Make folders and set parameters
os.makedirs('checkpoints', exist_ok=True)
save_images = True
best_losses = 1e10
epochs = 300

save_path = './ColorizationNetwork'
os.makedirs(save_path,exist_ok=True)
output_path = os.path.join(save_path, 'basic_model.tar')

# Train model
for epoch in range(epochs):
  # Train for one epoch, then validate
  #if epoch>110:
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0) #Optimizer
  train(train_dataloader, model, criterion, optimizer, epoch)
  
  with torch.no_grad():
    losses = validate(val_dataloader, model, criterion, save_images, epoch)
    print('loss: ',losses)
    scheduler.step(losses)
    
  # Save checkpoint and replace old best model if current model is better
  if losses < best_losses:
    best_losses = losses
    torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.6f}.pth'.format(epoch,losses))

###############################################################################################################
'''
os.makedirs('result', exist_ok=True)

def test_1epoch(net,dataloader):
  net.eval() 

  for i, data in enumerate(dataloader):
    if use_cuda:
      l = data["l"].to('cuda')
      hint = data["hint"].to('cuda')
      file_name = data['file_name']
    
    hint_image = torch.cat((l, hint), dim=1)
    output_ab = net(hint_image)

    for j in range(len(output_ab)): # save at most 5 images
      save_path = {'result': 'result/'}
      to_rgb_result(l[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=file_name[j])

PATH="checkpoints/model-epoch-54-losses-0.000321-scores-0.963141.pth"
model.load_state_dict(torch.load(PATH))

test_1epoch(model,test_dataloader)
'''
