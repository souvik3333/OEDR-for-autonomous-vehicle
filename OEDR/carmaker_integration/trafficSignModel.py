import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 1)       
        
        self.conv2 = nn.Conv2d(32, 64, (5,5))
                
        self.conv3 = nn.Conv2d(64, 128, (5,5))
        
        # input dim = 16*5*5, output dim = 120
        
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        
        # input dim = 120, output dim = 84
        
        self.fc2 = nn.Linear(1024, 512)
        
        # input dim = 84, output dim = 10
        
        self.fc3 = nn.Linear(512, 43)

    def forward(self, x):
        # pool size = 2
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        
        x = F.max_pool2d(x, 2)
        
#         x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # flatten as one dimension
        x = x.view(x.size()[0], -1)
        # input dim = 16*5*5, output dim = 120
        x = F.relu(self.fc1(x))
        # input dim = 120, output dim = 84
        x = F.relu(self.fc2(x))
        # input dim = 84, output dim = 10
        x = self.fc3(x)
        return x
