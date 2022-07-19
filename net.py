# NestFuse network 
class NestFuse(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(NestFuse, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()
        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        # self.DB5_0 = block(nb_filter[3], nb_filter[4], kernel_size, 1)
        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)

        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.cbam160= CBAM160(160,20)
        self.cbam112 = CBAM112(112,17)
        self.cbam64 = CBAM64(64,8)
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.CONVT1 = nn.Conv2d(160, 112, 3, padding=1, stride=1)
        self.CONVT2 = nn.Conv2d(112, 64, 3, padding=1, stride=1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)
# 编码器部分
    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]
    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))
        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))
        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))  # torch.Size([1, 160, 64, 64])
        # print(x3_1.shape)
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))   # torch.Size([1, 112, 128, 128])
        # print(x2_2.shape)
        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))  # torch.Size([1, 64, 256, 256])
        # print(x1_3.shape)
        x3_1 = self.cbam160(x3_1)
        x3_1 = self.CONVT1(x3_1)
        x2_2 = self.cbam112(x2_2) + self.Img_up(x3_1)
        x2_2 = self.CONVT2(x2_2)
        x1_3 = self.cbam64(x1_3)+self.Img_up(x2_2)

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]
        

  
