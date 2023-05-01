from msilib.schema import Binary
import torch
import os
import pandas as pd
import shutil
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import matplotlib
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import cv2
from torch.utils.data import Dataset, DataLoader
import argparse
matplotlib.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# Helper functions
def model_keypoints_plot(image, outputs, orig_keypoints, epoch, ind, split):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    
    # just get a single datapoint from each batch
    img = image[0]
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'rx')
        plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
    if epoch == None:
        plt.savefig(f"{output_path}/image_{ind}.png")
    else:
        plt.savefig(f"{output_path}/{split}_vis/epoch_{epoch}_batch_{ind}.png")
    plt.close()

def dataset_keypoints_plot(data):
    """
    This function shows the image faces and keypoint plots that the model
    will actually see. This is a good way to validate that our dataset is in
    fact corrent and the faces align wiht the keypoint features. The plot 
    will be show just before training starts. Press `q` to quit the plot and
    start training.
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(data)):
        if i == 9:
          break
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        keypoints = sample['keypoints']
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], 'b.')
    plt.show()
    plt.close()

def train_val_test_split(csv_path, split):
    cumulative_split = np.array([np.sum(split[:i]) for i in range(len(split))[1:]])
    df_data = pd.read_csv(csv_path, header=None)
    len_data = len(df_data)
    randInd = np.arange(0,len_data)
    np.random.seed(seed)
    np.random.shuffle(randInd)
    train_ind, val_ind, test_ind = np.split(randInd, np.round(cumulative_split*len_data, decimals=0).astype(int))
    train_samples = df_data.iloc[train_ind][:]
    val_samples = df_data.iloc[val_ind][:]
    test_samples = df_data.iloc[test_ind][:]

    return train_samples, val_samples, test_samples

def fit(model, dataloader, data, criterion, optimizer):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints, heatmap = data['image'].to(DEVICE), data['keypoints'].to(DEVICE), data['heatmap'].to(DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        # print(outputs.shape)
        # print(heatmap.shape)
        # print(outputs.view(-1,1).shape)
        # print(heatmap.view(-1,1).shape)
        loss = criterion(outputs.view(-1,1), heatmap.view(-1,1))
        if i>20:
            plt.imshow(outputs.detach().cpu().numpy()[0,0,:,:])
            plt.show()
        print(loss)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss

def evaluate(model, dataloader, data, epoch, criterion, split):
    print(f'Evaluating on {split}')
    model.eval()
    running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data['image'].to(DEVICE), data['keypoints'].to(DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if epoch == None: # do all images for epoch = None
                model_keypoints_plot(image, outputs, keypoints, epoch, i, split)
            elif (epoch+1) % 1 == 0 and i < 9: # only do first image of first 9 batches for each training epoch
                model_keypoints_plot(image, outputs, keypoints, epoch, i, split)
        
    loss = running_loss/counter
    return loss


def train(model):
  # optimizer
  optimizer = optim.Adam(model.parameters(), lr=LR)
  # we need a loss function which is good for regression like SmmothL1Loss ...
  # ... or MSELoss
  criterion = nn.SmoothL1Loss()

  train_loss = []
  val_loss = []
  real_loss = []
  best_real_loss = 10000
  for epoch in range(EPOCHS):
      print(f"Epoch {epoch+1} of {EPOCHS}")
      train_epoch_loss = fit(model, train_loader, train_data, criterion, optimizer)
      val_epoch_loss = evaluate(model, valid_loader, valid_data, epoch, criterion, 'val')
      real_epoch_loss = evaluate(model, real_loader, real_data, epoch, criterion, 'real')
      train_loss.append(train_epoch_loss)
      val_loss.append(val_epoch_loss)
      real_loss.append(real_epoch_loss)
      print(f"Train Loss: {train_epoch_loss:.4f}")
      print(f'Val Loss: {val_epoch_loss:.4f}')
      print(f'Real Loss: {real_epoch_loss:.4f}')
      print('\n')

      if (epoch+1) % 10 == 0:
        torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': criterion,
              }, f"{output_path}/ckpt/epoch_{epoch}.pth")
      if best_real_loss > real_epoch_loss:
        best_real_loss = real_epoch_loss
        torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': criterion,
              }, f"{output_path}/ckpt/best_real_epoch_{str(epoch).zfill(2)}.pth")
  # loss plots
  plt.figure(figsize=(10, 7))
  plt.plot(train_loss, color='orange', label='train loss')
  plt.plot(val_loss, color='red', label='validataion loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(f"{output_path}/loss.png")
  #plt.show()

  # loss plots for real
  plt.figure(figsize=(10, 7))
  plt.plot(real_loss, color='orange', label='real loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(f"{output_path}/real_loss.png")
  #plt.show()

  torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': criterion,
              }, f"{output_path}/ckpt/epoch_{epoch}.pth")

  print('DONE TRAINING')

# evaluation function
def test(model, dataloader, data, criterion):
    print('Testing')
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data['image'].to(DEVICE), data['keypoints'].to(DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            print(image.shape)
            print(outputs)
            if i < 10:
                test_keypoints_plot(image, outputs, keypoints, i)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

def test_keypoints_plot(image, outputs, orig_keypoints, i):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    
    # just get a single datapoint from each batch
    img = image[0]
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    #plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
        plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
      
    plt.savefig(f"{output_path}/test_image_{i}.png")
    plt.close()

class FaceKeypointDataset(Dataset):
    def __init__(self, samples, path):
        self.data = samples
        self.path = path
        self.resize = 112

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = cv2.imread(f"{self.path}/{self.data.iloc[index][0]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w, channel = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))
        # get the keypoints
        keypoints = self.data.iloc[index][1:]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
        
        heatmap = np.zeros((keypoints.shape[0], self.resize, self.resize))
        for idx, kp in enumerate(keypoints):
            kp = np.floor(kp).astype(int)
            for n in range(-1,2):
                for m in range(-1,2):
                    heatmap[idx, kp[1]+n, kp[0]+m] = 1
        # plt.subplot(2,3,1)
        # plt.imshow(heatmap[0,:,:],alpha=0.8)
        # plt.imshow(np.transpose(image, (1, 2, 0)),alpha=0.6)
        # plt.subplot(2,3,2)
        # plt.imshow(heatmap[1,:,:],alpha=0.8)
        # plt.imshow(np.transpose(image, (1, 2, 0)),alpha=0.6)
        # plt.subplot(2,3,3)
        # plt.imshow(heatmap[2,:,:],alpha=0.8)
        # plt.imshow(np.transpose(image, (1, 2, 0)),alpha=0.6)
        # plt.subplot(2,3,4)
        # plt.imshow(heatmap[3,:,:],alpha=0.8)
        # plt.imshow(np.transpose(image, (1, 2, 0)),alpha=0.6)
        # plt.subplot(2,3,5)
        # plt.imshow(heatmap[4,:,:],alpha=0.8)
        # plt.imshow(np.transpose(image, (1, 2, 0)),alpha=0.6)
        # plt.show()
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
            'heatmap': torch.tensor(heatmap, dtype=torch.float)
        }

class FaceKeypointResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(FaceKeypointResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layer
        self.l0 = nn.Linear(2048, 10)
        self.deconv = nn.ConvTranspose2d(in_channels=2048, out_channels=5, kernel_size=109, stride=1) #nn.ConvTranspose2d(in_channels=2048, out_channels=5, kernel_size=218, stride=1)
        # for i, layer in enumerate(self.model.children()):
        #   print(i, layer)
          #for sublayer in enumerate(layer.children()):
            #print(sublayer)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        #print(f'Model input: {x.shape}')
        x = self.model.features(x)
        out = self.deconv(x)
        #print(out.shape)
        #print(out[0,0,:,:].argmax().shape)
        #print(torch.argmax(out).shape)
        #print(f'ResNet output: {x.shape}')
        #x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        #print(f'Adapool output: {x.shape}')
        #l0 = self.l0(x)
        #print(f'FC output: {x.shape}')
        #print(l0.shape)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load training parameters.')
    parser.add_argument('--mode', type=str)
    # Training parameters
    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--real_dataset', type=str)
    parser.add_argument('--train_tag', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    # Evaluation parameters
    parser.add_argument('--eval_dataset', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--eval_tag', type=str)

    args = parser.parse_args()

    # Set randomness parameters
    print(f'Using seed {args.seed}')
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {DEVICE}')
    ########## Training ##########
    if str(args.mode) == 'train':
        print(f'Training on dataset: {args.train_dataset}')
        print(f'Output tag: {args.train_tag}')
        print(f'Using real dataset: {args.real_dataset}')
        # Configure paths
        root_path = args.train_dataset
        output_path = os.path.join(root_path, 'TrainingOutput', args.train_tag)

        # Learning params
        BATCH_SIZE = 32
        LR = 0.001
        EPOCHS = args.epochs


        # train/test split
        train_val_test = [0.7, 0.2, 0.1]

        # show dataset keypoint plot
        SHOW_DATASET_PLOT = False

        # make output directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path,'val_vis')):
            os.makedirs(os.path.join(output_path,'val_vis'))
        if not os.path.exists(os.path.join(output_path,'real_vis')):
            os.makedirs(os.path.join(output_path,'real_vis'))
        if not os.path.exists(os.path.join(output_path,'ckpt')):
            os.makedirs(os.path.join(output_path,'ckpt'))

        # get the training and validation data samples
        training_samples, valid_samples, test_samples = train_val_test_split(f"{root_path}/keypoints.csv", train_val_test)
        print(args.real_dataset)
        real_samples, _, _ = train_val_test_split(f"{args.real_dataset}\keypoints.csv", [1,0,0])

        # initialize the dataset - `FaceKeypointDataset()`
        for folder in os.listdir(root_path):
            if folder[:3]=='RGB':
                picfolder = folder
            if folder[:7]=='Dataset':
                jsonfolder = folder
        train_data = FaceKeypointDataset(training_samples, 
                                        f"{root_path}/{picfolder}")
        valid_data = FaceKeypointDataset(valid_samples, 
                                        f"{root_path}/{picfolder}")
        test_data = FaceKeypointDataset(test_samples, 
                                        f"{root_path}/{picfolder}")
        real_data = FaceKeypointDataset(real_samples, 
                                        f"{args.real_dataset}\RGB")

        # prepare data loaders
        train_loader = DataLoader(train_data, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True)
        valid_loader = DataLoader(valid_data, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False)
        test_loader = DataLoader(test_data, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False)
        real_loader = DataLoader(real_data, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False)
        print(f"Training sample instances: {len(train_data)}")
        print(f"Validation sample instances: {len(valid_data)}")
        print(f"Test sample instances: {len(test_data)}")
        print(f"Real sample instances: {len(real_data)}")

        if SHOW_DATASET_PLOT:
            dataset_keypoints_plot(valid_data)
        # model 
        model = FaceKeypointResNet50(pretrained=True, requires_grad=True).to(DEVICE)

        train(model)







    ########## Evaluation ##########
    if args.mode == 'eval':
        print(f'Evaluating on dataset: {args.eval_dataset}')
        print(f'Checkpoint used: {args.checkpoint}')
        # Load saved model
        model = FaceKeypointResNet50(pretrained=True, requires_grad=True).to(DEVICE)
        model.load_state_dict(torch.load(args.checkpoint,map_location=torch.device('cpu'))['model_state_dict'])
        model.eval()

        # Check performance on test data
        root_path = args.eval_dataset
        output_path = os.path.join(root_path, 'EvaluationOutput', args.eval_tag)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for folder in os.listdir(root_path):
            if folder[:3]=='RGB':
                picfolder = folder
            if folder[:7]=='Dataset':
                jsonfolder = folder

        testing_samples, _, _ = train_val_test_split(root_path + '/keypoints.csv', [1, 0, 0])

        test_data = FaceKeypointDataset(testing_samples, f"{root_path}/{picfolder}")
        # prepare data loaders
        test_loader = DataLoader(test_data, 
                                batch_size=1, 
                                shuffle=False)

        criterion = nn.SmoothL1Loss()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        test_loss = evaluate(model, test_loader, test_data, None, criterion, '')
        print(test_loss)



