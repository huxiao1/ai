import torch 
import torch.nn as nn

import os

from PIL import Image
from torchvision import transforms

tran = transforms.Compose([
    #transforms.Resize((112, 112), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

NUM_CLASSES = 10 

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True), # y=x+1, x=x+1
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding =1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*2*2)
        x = self.classifier(x)
        return x

def test():
    net = AlexNet().cuda()
    #x = torch.randn([1,3,32,32]).cuda()
    #y = net(x)
    #print(y) #training

    model_path = os.path.join('weights','alexnet.pt')
    print("Model:" + model_path)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])

    net=net.cuda()
    print(net)

    test_image = os.path.join('test.jpg')
    img = Image.open(test_image)
    img_tensor = tran(img)

    #print(img_tensor)
    img_tensor = img_tensor.unsqueeze_(0)
    print(img_tensor)
    input = img_tensor.cuda()

    y = net(input)

    print(y)


if __name__ == '__main__':
    test()


