
import os
import pandas as pd
from PIL import Image
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
import torch.nn as nn
from torchsummary import summary
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
import itertools
import matplotlib.pyplot as matplt
import warnings
from sklearn.metrics import classification_report, confusion_matrix




#variables

epochs = 15
learn_Rate = 0.001



# reading our datasets
data_Store = os.path.abspath("New_Datasets")

dataSet = pd.DataFrame()

modelLocn="/Users/arvindsangwan/Desktop/AI Project/model1.pth"
dir=["New_Datasets/without_mask","New_Datasets/cloth_mask","New_Datasets/surgical_mask","New_Datasets/N95_mask","New_Datasets/Incorrect_Mask"]

for i in range (0,len(dir)):
    for path in os.listdir(dir[i]):
        full_path = os.path.abspath(dir[i])
        dataSet=dataSet.append({'img': str(full_path+"/" + path),'groups': i},ignore_index=True)
        
storedObject = 'New_Datasets/dataset.pickle'
print(f'Saving Dataframe to: {storedObject}')
dataSet.to_pickle(storedObject)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#image transforming and resizing
class maskDetection(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

        self.transformations = Compose([
            Resize((32, 32)),
            ToTensor(),
            Normalize(
               mean=[0.5, 0.5, 0.5],
               std=[0.5, 0.5, 0.5]
            )
        ])
    def __len__(self):
        return len(self.dataFrame.index)

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('error: slicing not possible')
        val1 = self.dataFrame.iloc[key]
        val2 = Image.open(val1['img']).convert('RGB')
        return {
            'img': self.transformations(val2),
            'groups': tensor([val1['groups']], dtype=long),
            'path': val1['img']
        }


    
#CNN model
class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        #[(W−K+2P)/S]+1
        # W is the input volume
        # K is the Kernel size 
        # P is the padding 
        # S is the stride 
        #64
        #64-3+2+1= 64
        #(3,32,32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        #32-5+2+1=32
        #(20*32*32)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2,2)
        #16
        #(20*16*16)
        #16-3+2+1=16
        #32*16*16
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        #16-3+2+1=16
        #op=32
        #(32*16*16)
        self.fc1 = nn.Linear(32*16*16, 5)

    def forward(self, input):
        _out = nn.functional.relu(self.bn1(self.conv1(input)))      
        _out = nn.functional.relu(self.bn2(self.conv2(_out)))     
        _out = self.pool(_out)                            
        _out = nn.functional.relu(self.bn3(self.conv3(_out)))     
        _out = _out.view(-1, 32*16*16)
        _out = self.fc1(_out)

        return _out

def prepareData(path) -> None:
    maskData = pd.read_pickle(path)
    # print the distribution
    print(maskData['groups'].value_counts())
    train, test = train_test_split(maskData, test_size=0.25, random_state=0,
                                       stratify=maskData['groups'])
    return [
        maskDetection(train),
        maskDetection(test),
        nn.CrossEntropyLoss()
    ]

#plot
def matPlot(cmatrix, groups, normalize=False, heading='CONFUSION MATRIX', type_map=matplt.cm.viridis):
    matplt.imshow(cmatrix, cmap=type_map)
    lbl = np.arange(len(groups))
    matplt.xticks(lbl, groups, rotation=60)
    matplt.title(heading)
    matplt.colorbar()
    matplt.yticks(lbl, groups)

    print(cmatrix)

    threshold = cmatrix.max() / 4.
    for a, b in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        matplt.text(b, a, cmatrix[a, b],
                 horizontalalignment="center",
                 color="green" if cmatrix[a, b] > threshold else "yellow")

    matplt.tight_layout()
    matplt.ylabel('Actual label')
    matplt.xlabel('Predicted label')


cnn_modelData = CnnModel().to(device)

def loadTraniningData(train_df) -> DataLoader:
    return DataLoader(train_df, batch_size=64, shuffle=True, num_workers=0)


def trainModel():
    optimize =Adam(cnn_modelData.parameters(), lr=learn_Rate)
    for epoch in range(epochs):
        epoch_data_loss = 0.0
        train_loader=loadTraniningData(train_df)
        for i, data in enumerate(train_loader, 0):
            img, tar = data['img'], data['groups']
            img, tar = img.to(device),tar.to(device)
            labels = tar.flatten()
            outData = cnn_modelData(img)
            batchLoss = cross_entropy_loss(outData, labels)
            optimize.zero_grad() 
            batchLoss.backward()
            optimize.step()
            epoch_data_loss += batchLoss
        print(f' Training loss at {epoch}:',epoch_data_loss)


def testDataLoader(validate_df) -> DataLoader:
    return DataLoader(validate_df, batch_size=64, num_workers=0)


train_df, validate_df, cross_entropy_loss = prepareData("New_Datasets/dataset.pickle")


def save_model(modelLocn):
    torch.save(cnn_modelData.state_dict(), modelLocn)
    print("model_Saved")

save_model(modelLocn)  

def load_model(modelLocn):
    model=CnnModel()
    model.load_state_dict(torch.load(modelLocn))
    model.eval()
    print("model_loaded")
    return model


def evaluation():
    data_prediction, real_Data = torch.tensor([]), torch.tensor([])
    train_loader=testDataLoader(validate_df)
    for i, data in enumerate(train_loader):
        images, targets = data['img'], data['groups']
        if device.type=='cuda':
            images = images.cuda()
            targets = targets.cuda()
        targets = targets.flatten()
        _out = cnn_modelData(images)
        _out = torch.argmax(_out, axis=1)
        if device.type == 'cuda':
            data_prediction = data_prediction.cuda()
            real_Data = real_Data.cuda()
        data_prediction = torch.cat((data_prediction, _out.flatten()), dim=0)
        real_Data = torch.cat((real_Data, targets), dim=0)

        # print metrics
    groups = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Incorrect_Mask']

    print(classification_report(real_Data.cpu(), data_prediction.cpu(), digits=2, target_names=groups))
    confusion_mat = confusion_matrix(real_Data.cpu().numpy(), data_prediction.cpu().numpy())
    matPlot(confusion_mat, groups)



# printing the summary
print(summary(cnn_modelData, input_size=(3, 32, 32)))



trainModel()
print('model has been trained')

evaluation()



print("Current Directory Path :: ===>" + os.getcwd())







imgpath="//Users/arvindsangwan/Desktop/AI Project/New_Datasets/N95_mask/N95_mask 5.jpg"
im="/Users/arvindsangwan/Desktop/AI Project/New_Datasets/without_mask/without_mask 2.jpg"
im2="/Users/arvindsangwan/Desktop/AI Project/New_Datasets/surgical_mask/sugical_mask 4.jpg"
im3="/Users/arvindsangwan/Desktop/AI Project/New_Datasets/cloth_mask/cloth_mask 1.jpg"
im4="/Users/arvindsangwan/Desktop/AI Project/New_Datasets/Incorrect_Mask/Incorrect_Mask 27.jpg"


def predict(imgpath):
    newDataFrame = pd.DataFrame()
    #imgpath="New_Datasets\N95_mask\\N95_mask (1).png"
    newDataFrame=newDataFrame.append({'img': str(imgpath),'groups': 1},ignore_index=True)
    newDataFrame=maskDetection(newDataFrame)
    out_classes = ['without_mask', 'with_mask_surgical', 'with_mask_cloth', 'N95_mask', 'Incorrect_Mask']
    loaded_trainData=loadTraniningData(newDataFrame)
    for i, data in enumerate(loaded_trainData, 0):
        print(data)
        img, tar = data['img'], data['groups']
        #img, tar = img.to(device),tar.to(device)
        labels = tar.flatten()
        model=load_model(modelLocn)
        outputs = model(img) 
        _out = torch.argmax(outputs, axis=1)    
        print("Predicted group is ",out_classes[_out[0]])


predict(imgpath)
predict(im)
predict(im2)
predict(im3)
predict(im4)









