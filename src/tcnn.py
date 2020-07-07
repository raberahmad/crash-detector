import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
from pathlib import Path
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict



class TCNN:
    def __init__(self):
        self.project_root_dir = Path(__file__).parents[1]
        self.data_dir = str(self.project_root_dir) + "/dataset-beamng-cropped-with-real-test"
        
        self.train_dir = self.data_dir + "/train"
        self.valid_dir = self.data_dir + "/valid"
        self.test_dir = self.data_dir + "/test"

    def prepare_data(self):
        # tranform for training, validation and testing are being defined.
        training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

        validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

        testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])

        # dataset is being loaded
        self.training_dataset = datasets.ImageFolder(self.train_dir, transform=training_transforms)
        self.validation_dataset = datasets.ImageFolder(self.valid_dir, transform=validation_transforms)
        self.testing_dataset = datasets.ImageFolder(self.test_dir, transform=testing_transforms)

        # dataloaders are defined with a specific batchsize 
        self.train_loader = torch.utils.data.DataLoader(self.training_dataset, batch_size=30, shuffle=True)
        self.validate_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=14)
        self.test_loader = torch.utils.data.DataLoader(self.testing_dataset, batch_size=14)




    def load_model(self, file_path):
        
        loaded = torch.load(str(self.project_root_dir)+"/saved-models/"+file_path+".pth", map_location=torch.device('cpu'))

        # if loaded['arch'] == "vgg16-aug" or "vgg16-simple" or "vgg16-simple-2" or "vgg16-full" or "vgg16-cropped":

        #     self.model = models.vgg16(pretrained=True)

        #     classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
        #                                 ('relu', nn.ReLU()),
        #                                 ('drop', nn.Dropout(p=0.5)),
        #                                 ('fc2', nn.Linear(5000, 2000)),
        #                                 ('relu', nn.ReLU()),
        #                                 ('drop',nn.Dropout(p=0.5)),
        #                                 ('fc3', nn.Linear(2000, 2)),
        #                                 ('output', nn.LogSoftmax(dim=1))]))

        #     for param in self.model.parameters():
        #         param.requires_grad = False
        
        # if loaded['arch'] == "vgg16-aug-2" :
            
        #     self.model = models.vgg16_bn(pretrained=True)

        #     classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
        #                                 ('relu', nn.ReLU()),
        #                                 ('drop', nn.Dropout(p=0.5)),
        #                                 ('fc2', nn.Linear(5000, 2000)),
        #                                 ('relu', nn.ReLU()),
        #                                 ('drop',nn.Dropout(p=0.5)),
        #                                 ('fc3', nn.Linear(2000, 1000)),
        #                                 ('relu', nn.ReLU()),
        #                                 ('drop',nn.Dropout(p=0.5)),
        #                                 ('fc4', nn.Linear(1000, 2)),
        #                                 ('output', nn.LogSoftmax(dim=1))]))
            
        #     for param in self.model.parameters():
        #         param.requires_grad = False
                
        if loaded['arch'] == "alexnet-test" or "alexnet-test-full" or "alexnet-test-cropped" or "alexnet-cropped-extra":
            
            self.model = models.alexnet(pretrained=True)

            classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(9216, 4096)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(4096, 500)),
                                        ('relu', nn.ReLU()),
                                        ('drop',nn.Dropout(p=0.5)),
                                        ('fc3', nn.Linear(500, 2)),
                                        ('output', nn.LogSoftmax(dim=1))]))
            
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("Model not found....")
        
        self.model.class_to_idx = loaded['class_to_idx']  
    
        self.model.classifier = classifier



        self.model.load_state_dict(loaded['model_state_dict'])

        return self.model



    def save_model(self, model):
        self.model.class_to_idx = self.training_dataset.class_to_idx

        checkpoint = {'arch': self.model_name,
                        'class_to_idx': model.class_to_idx,
                        'model_state_dict': model.state_dict()
                        }

        torch.save(checkpoint, str(self.project_root_dir)+"/saved-models/"+self.model_name+'.pth')



    def test_accuracy(self, model):
        self.model.eval()
        self.model.to('cpu')

        with torch.no_grad():

            accuracy = 0

            for images, labels in iter(self.test_loader):

                images, labels = images.to('cpu'), labels.to('cpu')

                output = self.model.forward(images)

                probabilities = torch.exp(output)
            
                equality = (labels.data == probabilities.max(dim=1)[1])
            
                accuracy += equality.type(torch.FloatTensor).mean()

        print("Test Accuracy: {}".format(accuracy/len(self.test_loader)))


    def predict_class(self, img_frame):

        #converting to pytorch tensor
        img_frame = torch.from_numpy(img_frame).type(torch.FloatTensor)
        

        img_frame = img_frame.unsqueeze(0)

        output = self.model.forward(img_frame)
        probabilities = torch.exp(output)
        topk2, top_indices = probabilities.topk(2)

        #converting to list
        topk2 = topk2.detach().type(torch.FloatTensor).numpy().tolist()[0] 
        top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 

        idx_to_class = {value: key for key, value in self.model.class_to_idx.items()}
        #print(idx_to_class)
    
        top_classes = [idx_to_class[index] for index in top_indices]
    
        return topk2, top_classes

    


    

