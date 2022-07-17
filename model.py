
class DUAMNet(nn.Module):
    def __init__(self, args):
        super(DUAMNet, self).__init__()
        self.T1 = DUAM_BU(args)
        self.T2 = DUAM_BU(args)
        self.T3 = DUAM_BU(args)
        self.T4 = DUAM_BU(args)
        self.cbam = CBAM(96, 16)

    def forward(self, x_input):
        x1 = x_input  
        lbp1 = get_LBP(x1)
        x1 = torch.cat((x1, lbp1), 1)  
        res_g1_s1 = self.T1(x1)

        x2 = res_g1_s1
        lbp2 = get_LBP(x2)
        x2 = torch.cat((x2, lbp2), 1)
        res_g2_s1= self.T2(x2)

        x3 = res_g2_s1
        lbp3 = get_LBP(x3)
        x3 = torch.cat((x3, lbp3), 1) 
        res_g3_s1 = self.T3(x3)

        x4 = res_g3_s1
        lbp4 = get_LBP(x4)
        x4= torch.cat((x4, lbp4), 1)  # 96
        res_g4_s1 = self.T4(x4)
        return res_g4_s1, res_g4_s2, res_g4_s4

class DUAM_BU(nn.Module):
    def __init__(self, args):
        super(DUAM_BU, self).__init__()
        K = 16
        kSize = 3
        self.D = 6
        G = 8
        C = 4
        self.SFENet1 = nn.Conv2d(K, K, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(K, K, kSize, padding=(kSize - 1) // 2, stride=1)
        self.RDBs = nn.ModuleList()
        self.DUAMS.append(
            DUAM(g0=K, g1=G, nConvLayers=C)
        )
        self.DUAMS.append(
            DUAM(g0=K, g1=G, nConvLayers=C)
        )
        self.DUAMS.append(
            DUAM(g0=2 * G0, g1=2 * G, nConvLayers=C)
        )
        self.DUAMS.append(
            DUAM(g0=2 * K, g1=2 * G, nConvLayers=C)
        )
        self.DUAMS.append(
            DUAM(g0=K, g1=G, nConvLayers=C)
        )
        self.DUAMS.append(
          DUAM(g0=K, g1=G, nConvLayers=C)
        )
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(K, K, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(K, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])
        self.UPNet2 = nn.Sequential(*[
            nn.Conv2d(K, K, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(K, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.UPNet4 = nn.Sequential(*[
            nn.Conv2d(G0 * 2, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0 * 2, kSize, padding=(kSize - 1) // 2, stride=2)

        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize + 1, stride=2, padding=1)
        self.Up2 = nn.ConvTranspose2d(G0 * 2, G0, kSize + 1, stride=2, padding=1)

        self.Relu = nn.ReLU()
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.CONVT1 = nn.Conv2d(16, 1, 3, padding=1, stride=1)

    def part_forward(self, x):
        flag = x[0]
        input_x = x[1] 
        input_nc = 1
        output_nc = 1
        deepsupervision = False
        nb_filter = [64, 112, 160, 208, 256]

        # AU-Net++: inner-encoder
        nest_model = NestFuse(nb_filter, input_nc, output_nc, deepsupervision)
        nest_model = nest_model.cuda()
        input_x1 = self.CONVT1(input_x)
        en = nest_model.encoder(input_x1)
        # AU-Net++: inner-decoder
        outputs = nest_model.decoder_train(en)

       # U-Net:outer-encoder
        first = self.Relu(self.SFENet1(input_x))  
        f_s1 = self.Relu(self.SFENet2(f_first)) 
        f_s2 = self.Down1(self.DUAMS[0](s1)) 
        f_s4 = self.Down2(self.DUAMS[1](s2))  

        # U-Net: outer-decoder
        if flag == 0:
            f_s4 = f_s4 + self.DUAMS[3](self.RDBs[2](f_s4))  
            f_s2 = f_s2 + self.DUAMS[4](self.Up2(f_s4))
            f_s1 = f_s1 + self.DUAMS[5](self.Up1(f_s2)) 
        else:
            f_s4 = f_s4 + self.DUAMS[3](self.RDBs[2](f_s4))
            f_s2 = f_s2 + self.DUAMS[4](self.Up2(f_s4))
            f_s1 = f_s1 + self.DUAMS[5](self.Up1(f_s2))

        res4 = self.UPNet4(f_s4)
        res2 = self.UPNet2(f_s2) + self.Img_up(res4)
        res1 = self.UPNet(f_s1) + self.Img_up(res2)
        res1 = res1 + outputs[0]
        return res1, res2, res4

    def forward(self, x_input):
        x = x_input
        res1, res2, res4= self.part_forward(x)
        return res1, res2, res4
