# def create_model(self, model_name):
    #     self.model_name = model_name

    #     self.model = models.vgg16(pretrained=True)

    #     for parameter in self.model.parameters():
    #         parameter.require_grad = False
        
    #     classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
    #                                     ('relu', nn.ReLU()),
    #                                     ('drop', nn.Dropout(p=0.5)),
    #                                     ('fc2', nn.Linear(5000, 2000)),
    #                                     ('relu', nn.ReLU()),
    #                                     ('drop',nn.Dropout(p=0.5)),
    #                                     ('fc3', nn.Linear(2000, 2)),
    #                                     ('output', nn.LogSoftmax(dim=1))]))

    #     self.model.classifier = classifier

        



    # def validation(self, criterion):
    #     val_loss = 0
    #     accuracy = 0

    #     for images, labels in iter(self.validate_loader):
    
    #         images, labels = images.to('cuda'), labels.to('cuda')

    #         output = self.model.forward(images)
    #         val_loss += criterion(output, labels).item()

    #         probabilities = torch.exp(output)
            
    #         equality = (labels.data == probabilities.max(dim=1)[1])
    #         accuracy += equality.type(torch.FloatTensor).mean()
    
    #     return val_loss, accuracy

    # def train_model(self):
    #     criterion = nn.NLLLoss()

    #     optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)

    #     epochs = 100

    #     self.model.to('cuda')

    #     for e in range(epochs):
      
    #         self.model.train()

    #         running_loss = 0
    #         self.save_model(self.model)
            
    #         for images, labels in iter(self.train_loader):


    #             images, labels = images.to('cuda'), labels.to('cuda')

    #             optimizer.zero_grad()

    #             output = self.model.forward(images)
    #             loss = criterion(output, labels)
    #             loss.backward()
    #             optimizer.step()

    #             running_loss += loss.item()


    #             self.model.eval()
            
    #             with torch.no_grad():
    #                 validation_loss, accuracy = self.validation(criterion)

    #             print("Epoch: {}/{}.. ".format(e+1, epochs),
    #                     "Validation Loss: {:.3f}.. ".format(validation_loss/len(self.validate_loader)),
    #                     "Validation Accuracy: {:.3f}".format(accuracy/len(self.validate_loader)))

    #             running_loss = 0
    #             self.model.train()

    #             torch.cuda.empty_cache()