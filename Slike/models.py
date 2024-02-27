import torch 
import torch.nn as nn 

class CRAFT(nn.Module) : 

    def __init__(self) : 

        super(CRAFT , self).__init__()

        self.conv_1 = nn.Conv2d(3 , 64 , 3 , 2 , 1)
        self.conv_2 = nn.Conv2d(64 , 128 , 3 , 2 , 1)
        self.conv_3 = nn.Conv2d(128 , 256 , 3 , 2 , 1)
        self.conv_4 = nn.Conv2d(256 , 512 , 3 , 2 , 1)
        self.conv_5 = nn.Conv2d(512 , 512 , 1 , 1 , 0)
        self.conv_6 = nn.Conv2d(32 , 32 , 3 , padding = 1)
        self.conv_7 = nn.Conv2d(32 , 32 , 3 , padding = 1)
        self.conv_8 = nn.Conv2d(32 , 16 , 3 , padding = 1)
        self.conv_9 = nn.Conv2d(16 , 16 , 3 , padding = 1)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(512 , 512 , 1) , 
            nn.BatchNorm2d(512) , 

            nn.Conv2d(512 , 256 , 3 , padding = 1) , 
            nn.BatchNorm2d(256) , 

            nn.Upsample(scale_factor = 2)
        )

        self.up_sample_2 = nn.Sequential(
            nn.Conv2d(256 , 256 , 1) , 
            nn.BatchNorm2d(256) , 

            nn.Conv2d(256 , 128 , 3 , padding = 1) , 
            nn.BatchNorm2d(128) , 

            nn.Upsample(scale_factor = 2)
        )

        self.up_sample_3 = nn.Sequential(
            nn.Conv2d(128 , 128 , 1) , 
            nn.BatchNorm2d(128) , 

            nn.Conv2d(128 , 64 , 3 , padding = 1) , 
            nn.BatchNorm2d(64) , 

            nn.Upsample(scale_factor = 2)
        )

        self.up_conv_1 = nn.Sequential(
            nn.Conv2d(64 , 64 , 1) , 
            nn.BatchNorm2d(64) , 

            nn.Conv2d(64 , 32 , 3 , padding = 1) , 
            nn.BatchNorm2d(32) 
        )

    def forward(self , image) : 

        conv1 = self.conv_1(image) # (h x w x 3) -> (h/2 x w/2 x 64)
        conv2 = self.conv_2(conv1) # (h/2 x w/2 x 64) -> (h/4 x w/4 x 128)
        conv3 = self.conv_3(conv2) # (h/4 x w/4 x 128) -> (h/8 x w/8 x 256)
        conv4 = self.conv_4(conv3) # (h/8 x w/8 x 256) -> (h/16 x w/16 x 512)
        conv5 = self.conv_5(conv4) # (h/16 x w/16 x 512) -> (h/32 x w/32 x 512)

        upsample_1 = self.up_sample_1(
            conv4 + conv5
        ) # (h/32 x w/32 x 512) -> (h/16 x w/16 x 256)
        
        upsample_2 = self.up_sample_2(
            upsample_1 + conv3
        ) # (h/16 x w/16 x 256) -> (h/8 x w/8 x 128)

        upsample_3 = self.up_sample_3(
            upsample_2 + conv2
        ) # (h/8 x w/8 x 128) -> (h/4 x w/4 x 64)

        upconv1 = self.up_conv_1(
            upsample_3 + conv1
        ) # (h/4 x w/4 x 64) -> (h/2 x w/2 x 32)

        conv6 = self.conv_6(upconv1) # (h/2 x w/2 x 32) -> (h/2 x w/2 x 32)
        conv7 = self.conv_7(conv6) # (h/2 x w/2 x 32) -> (h/2 x w/2 x 32)
        conv8 = self.conv_8(conv7) # (h/2 x w/2 x 32) -> (h/2 x w/2 x 16)
        conv9 = self.conv_9(conv8) # (h/2 x w/2 x 16) -> (h/2 x w/2 x 16)
        
        return conv9