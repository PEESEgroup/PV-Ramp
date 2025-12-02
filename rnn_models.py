import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from functools import partial
from einops import rearrange, reduce, repeat


class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1',nn.GroupNorm( 7 ,F_hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(3,3),
                              padding=(1,1), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]      
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)        # prediction
        next_hidden = hidden_tilde + K * (x-hidden_tilde)   # correction , Haddamard product     
        return next_hidden


class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        
        return self.H , self.H 
    
    def initHidden(self,batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device) )

    def setHidden(self, H):
        self.H = H

        
class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):              
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]          
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [],[]   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        
        return (self.H, self.C), self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
            self.C.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=min(16, nout), num_channels=nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        output_padding = 1 if stride == 2 else 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class encoder_E(nn.Module):
    def __init__(self, nc=4, nf=32):
        super(encoder_E, self).__init__()
        self.c0 = nn.Sequential(
            nn.Conv2d(nc, nf, kernel_size=1),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.c1 = dcgan_conv(nf, nf, stride=2)
        self.c2 = dcgan_conv(nf, nf, stride=1)
        self.c3 = dcgan_conv(nf, 2 * nf, stride=2)

    def forward(self, x):
        h0 = self.c0(x)
        h1 = self.c1(h0)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h0, h2, h3


class decoder_D(nn.Module):
    def __init__(self, nc=3, nf=32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1)
        self.upc2 = dcgan_upconv(nf, nf, stride=2)
        self.upc3 = dcgan_upconv(nf * 2, nf, stride=1)
        self.upc4 = nn.Conv2d(nf, nc, kernel_size=1)

        self.chan_attn = ChannelAttention(nf * 2)

    def forward(self, x, skip):
        d1 = self.upc1(x)
        d2 = self.upc2(d1)
        d2_c = torch.cat([d2, skip], dim=1)
        d2_c = d2_c * self.chan_attn(d2_c)
        d3 = self.upc3(d2_c)
        return self.upc4(d3)         


class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1)
        self.c2 = dcgan_conv(nf, nf, stride=1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x


class decoder_specific(nn.Module):
    def __init__(self, nc=32, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1)
        self.upc2 = dcgan_upconv(nf, nc, stride=2)
        self.upc3 = dcgan_upconv(nf, nc, stride=1)

        self.chan_attn = ChannelAttention(64)

    def forward(self, x, skip):
        d1 = self.upc1(x) 
        d2 = self.upc2(d1) 
        d2_c = torch.cat([d2, skip], dim=1)
        d2_c = d2_c * self.chan_attn(d2_c)
        d3 = self.upc3(d2_c)
        return d3


class PhyDNet(nn.Module):
    def __init__(self, phycell, convcell, device):
        super(PhyDNet, self).__init__()
        self.encoder_E = encoder_E()
        self.encoder_Ep = encoder_specific()
        self.encoder_Er = encoder_specific()
        self.decoder_Dp = decoder_specific()
        self.decoder_Dr = decoder_specific()
        self.decoder_D = decoder_D()

        self.phycell = phycell.to(device)
        self.convcell = convcell.to(device)
        self.device = device


    def forward(self, input_img, first_timestep=False, decoding=False):

        skip1, skip2, x = self.encoder_E(input_img) 

        physics_feature = None if decoding else self.encoder_Ep(x) 
        residue_feature = self.encoder_Er(x)  

        _, hout_p = self.phycell(physics_feature, first_timestep) 
        hcout_r, hout_r = self.convcell(residue_feature, first_timestep) 

        phy_hout = hout_p[-1]
        conv_hout = hout_r[-1]
        conv_cout = hcout_r[1][-1]

        decoded_Dp = self.decoder_Dp(phy_hout, skip2)
        decoded_Dr = self.decoder_Dr(conv_hout, skip2)

        concat = decoded_Dp + decoded_Dr   
        output_img = torch.sigmoid( self.decoder_D(concat, skip1))

        return output_img


def create_gaussian_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = g[:, None] @ g[None, :]
    window_2d = window_2d / window_2d.sum()
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    assert img1.size() == img2.size(), "Input images must have the same dimensions"
    B, C, H, W = img1.size()
    window = create_gaussian_window(window_size, C).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # [B]


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1 - ssim(img1, img2, self.window_size, self.size_average)


class FrameLoss(nn.Module):
    def __init__(self, alpha: float = 0.84, window_size: int = 11, size_average: bool = True):
        super(FrameLoss, self).__init__()
        self.alpha = alpha
        self.ssim_loss = SSIMLoss(window_size, size_average)
        self.l1_loss = nn.L1Loss()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        ssim_val = self.ssim_loss(img1, img2)
        l1_val = self.l1_loss(img1, img2)
        return self.alpha * ssim_val + (1 - self.alpha) * l1_val
    




    