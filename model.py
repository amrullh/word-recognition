import torch
import torch.nn as nn
import torch.nn.functional as F

class model_wordRec(nn.Module):
    def __init__(self, img_h = 80,img_channels = 3, num_classes = 27, hidden_state = 128):
        super(model_wordRec, self).__init__()

        
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True)
        )

        self.rnn = nn.Sequential(
            nn.LSTM(2048, hidden_state, bidirectional = True, num_layers = 2, batch_first = False)    
        )
        self.embedding = nn.Linear(hidden_state *2, num_classes)

    def forward(self, x):

        conv = self.cnn(x)

        batch, channel, h, w = conv.size()
        assert h == 4, "tinggi feature map harus 4 agar bisa flatten"
        conv = conv.permute(3, 0 , 1, 2)
        conv = conv.view(w, batch, channel * h)

        #LSTM
        recurrent, _ = self.rnn(conv)

        output = self.embedding(recurrent)

        return output