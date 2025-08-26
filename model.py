import torch
import torch.nn as nn

class SRB(nn.Module):
    def __init__(self, in_channel):
        super(SRB, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.bn4 = nn.BatchNorm2d(in_channel)

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 5), padding=(0, 2))
        self.conv51 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(5, 1), padding=(2, 0))


        self.out = nn.Conv2d(in_channels=in_channel*3, out_channels=in_channel, kernel_size=1)

    def forward(self, inp):
        B, C, H, W = inp.shape

        x1 = self.bn1(inp)
        x2 = self.bn2(inp)
        x3 = self.bn3(inp)

        x1 = self.conv(x1)
        x2 = self.conv15(x2)
        x3 = self.conv51(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.out(x)

        x = self.bn4(x)
        x = x + inp

        return x

class FGM(nn.Module):
    def __init__(self, in_channel):
        super(FGM, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.bn3 = nn.BatchNorm2d(in_channel)

        self.dil_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, dilation=2,padding=2, stride=2)

        self.dep_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1)

    def forward(self, inp):


        y = self.bn1(inp)
        y = self.dil_conv(y)
        weight = self.pool(y)
        
        x = self.bn2(inp)
        x = self.dep_conv(x)

        x = x * weight
        x = self.bn3(x)
        x = self.conv(x)
        x = nn.ReLU()(x)

        x = x * inp
        x = x + inp # 在呈上權重
        return x 

class CEFF(nn.Module):
    def __init__(self, in_channel):
        super(CEFF, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1)

    def forward(self, inp):

        x = self.bn2(inp)
        x = self.conv3(x)
        x = nn.Sigmoid()(x)

        y = self.bn1(inp)
        y = self.conv1(y)

        x = y * x

        return x + inp
    

class DK(nn.Module): 
    def __init__(self, width):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(width)
        self.down = nn.MaxPool2d(2, 2)


        self.dilated_conv = nn.Conv2d(in_channels=width,out_channels=width,kernel_size=3,
                                      padding=2, dilation=2,bias=True)
        
        self.bn2 = nn.BatchNorm2d(width)
        self.Tanh = nn.Tanh()

        self.ceff = CEFF(width)

        # self.pool = nn.MaxPool2d(64)
        self.down_C = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1)


    def forward(self, inp, wieght):
        B, C, H, W = inp.shape
        wieght = 1 - wieght

        x = self.bn1(inp)
        x = self.down(x)
        x = self.dilated_conv(x)
        y = self.bn2(x)
        y = self.Tanh(y)

        x = x * y

        
        x = self.ceff(x)

        # x = self.pool(x)
        x = self.down_C(x)
        x = x.mean(dim=(2, 3), keepdim=True)
        
        x = x * wieght
        x = x * inp
        return x
    
class Block(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        

        self.sca = FGM(in_channel=c)

        self.sg = CEFF(c)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sca(x)
        x = self.sg(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        # x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class model(nn.Module):

    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        

        self.feature_pick = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=4, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, padding=0, stride=2, groups=1, bias=True),
            nn.ReLU(),
            FGM(4),
            nn.Conv2d(in_channels=4, out_channels=width, kernel_size=1, padding=0, groups=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Tanh()
        )
        self.DK = DK(width)


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.srbs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[Block(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            self.srbs.append(
                SRB(chan)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[Block(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[Block(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape

        x = self.intro(inp)

        weight = self.feature_pick(inp)
        z = self.DK(x, weight)
        x = x * weight

        encs = []
        for encoder, down, srb in zip(self.encoders, self.downs, self.srbs):
        # for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            y = srb(x)
            y = x
            encs.append(y)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = x + z
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]
    
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

if __name__ == '__main__':
    img_channel = 3
    width = 48

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = model(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)



    inp_shape = (2, 3, 256, 256)
    random_image = torch.rand(inp_shape, dtype=torch.float32)
    output = net(random_image)
    if torch.isnan(output).any():
        print("Output contains NaN values")
    else:
        print("Output does not contain NaN values")

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('FLOPs:', flops)
    print('Parameters:', params)
