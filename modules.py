import torch
import torch.nn as nn
import setting_2 as fst
import numpy as np
import utils as ut
import setting as st
from torch.autograd import Function
import torch.nn.functional as F



class InputDependentCombiningWeights(nn.Module):
    def __init__(self, in_plance, spatial_rank= 1):
        super(InputDependentCombiningWeights, self).__init__()
        """ 1 """
        self.dim_reduction_layer = nn.Conv3d(in_plance, spatial_rank, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        """ 2 """
        self.dilations = [1, 2, 4, 8]
        self.multiscale_layers = nn.ModuleList([])
        for i in range(len(self.dilations)):
            self.multiscale_layers.append(nn.Conv3d(spatial_rank, spatial_rank, kernel_size=3, stride=1, padding=0, dilation=self.dilations[i], groups=spatial_rank, bias=False))

        """ 3 """
        self.squeeze_layer = nn.Sequential(
            nn.Conv3d(spatial_rank * (len(self.dilations) + 2), spatial_rank * 3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
        )

        """ 4 """
        self.excite_layer = nn.Sequential(
            nn.Conv3d(spatial_rank * 3, spatial_rank * 6, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.Sigmoid(),
        )

        """ 5 """
        self.proj_layer = nn.Conv3d(spatial_rank * 6, spatial_rank, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, input_tensor, size):
        x_lowd = self.dim_reduction_layer(input_tensor)  # batch, 16, 85, 105, 80
        x_pool = nn.AvgPool3d(kernel_size=x_lowd.size()[-3:], stride=1)(x_lowd)

        x_multiscale = [
            F.interpolate(x_lowd, size=size, mode='trilinear', align_corners=True),
            F.interpolate(x_pool, size=size, mode='trilinear', align_corners=True),
        ]

        for r, layer in zip(self.dilations, self.multiscale_layers):
            x_multiscale.append(
                F.interpolate(layer(x_lowd), size=size, mode='trilinear', align_corners=True),
            )

        x_multiscale = torch.cat(x_multiscale, 1)
        x_0 = self.squeeze_layer(x_multiscale)
        x_0 = self.excite_layer(x_0)
        x_0 = self.proj_layer(x_0)

        return x_0

class Input_Dependent_LRLC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='relu', bn=True, bias=False, np_feature_map_size = None, n_K = 1):
        super(Input_Dependent_LRLC, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.Conv3d(in_channels, out_channels * n_K, kernel_size, stride, padding, dilation, groups=groups, bias=bias)

        self.bn = nn.BatchNorm3d(out_channels) if bn else None
        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        self.combining_weights_layer = InputDependentCombiningWeights(in_channels, spatial_rank=n_K)

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ zero initialized bias vectors for width, height, depth"""
        self.list_parameter_b = nn.ParameterList([])
        for i in range(self.dim_feature_map):
            alpha = nn.Parameter(torch.zeros(np_feature_map_size[i]))
            self.list_parameter_b.append(alpha)
        alpha = nn.Parameter(torch.zeros(out_channels))
        self.list_parameter_b.append(alpha)

    def forward(self, input_tensor):
        """ weight """
        out = self.cnn_layers(input_tensor) # batch, out * rank, w, h, d (6, 64, 42, 52, 39)
        batch_dim = out.shape[0]
        x_dim = out.shape[2]
        y_dim = out.shape[3]
        z_dim = out.shape[4]
        weight = self.combining_weights_layer(input_tensor, size=(x_dim, y_dim, z_dim)) # batch, n_K, 42, 52, 39
        out = out.view(batch_dim, self.out_channels, self.n_K, x_dim, y_dim, z_dim) # batch, f, n_K, w, h, d
        weight = weight.unsqueeze(1)  # batch, 1, n_K, w, h, d
        f_out = torch.sum((out * weight), axis = 2) # batch, f, w, h, d

        """ bias """
        xx_range = self.list_parameter_b[0]
        xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, x_dim)
        xx_range = xx_range[:, None, :, None, None]

        yy_range = self.list_parameter_b[1]
        yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, y_dim)  # (batch, 200)
        yy_range = yy_range[:, None, None, :, None]

        zz_range = self.list_parameter_b[2]
        zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, z_dim)  # (batch, 200)
        zz_range = zz_range[:, None, None, None, :]

        ww_range = self.list_parameter_b[3] # [a]
        ww_range = ww_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, self.out_channels)  # (batch, 200)
        ww_range = ww_range[:, :, None, None, None]

        f_out = f_out + xx_range + yy_range + zz_range + ww_range
        f_out = self.bn(f_out)
        f_out = self.act_func(f_out)
        return f_out

class LRLC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='relu', bn=True, bias=False, np_feature_map_size = None, n_K = 1):
        super(LRLC, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.ModuleList([])
        for i in range(n_K):
            self.cnn_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias))

        self.bn = nn.BatchNorm3d(out_channels) if bn else None
        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ one initialized weight vectors for width, height, depth"""
        self.list_K = nn.ParameterList([])
        for i in range(self.dim_feature_map):
            alpha = nn.Parameter(torch.ones(self.n_K, np_feature_map_size[i]))
            self.list_K.append(alpha)

        """ zero initialized bias vectors for width, height, depth"""
        self.list_parameter_b = nn.ParameterList([])
        for i in range(self.dim_feature_map):
            alpha = nn.Parameter(torch.zeros(np_feature_map_size[i]))
            self.list_parameter_b.append(alpha)
        alpha = nn.Parameter(torch.zeros(out_channels))
        self.list_parameter_b.append(alpha)

    def forward(self, input_tensor):
        """ weight """
        f_out = torch.zeros(list(input_tensor.shape[:1]) + [self.out_channels] + list(self.shape_feature_map), dtype=torch.float32).cuda()
        for i in range(self.n_K):
            out = self.cnn_layers[i](input_tensor)
            w_dim = out.shape[1]  # 44
            x_dim = out.shape[2]  # 44
            y_dim = out.shape[3]  # 54
            z_dim = out.shape[4]  # 41
            batch_size_tensor = out.shape[0]
            xx_ones = torch.ones([batch_size_tensor, z_dim], dtype=torch.float32).cuda()
            xx_ones = xx_ones[:, :, None]  # batch, z_dim, 1 (6, 41, 1)
            xx_range = self.list_K[0][i]
            xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)
            xx_range = xx_range[:, None, :] # batch, 1, x_dim
            xx_channel = torch.matmul(xx_ones, xx_range) # batch, z_dim, x_dim
            xx_channel = xx_channel.unsqueeze(3).repeat(1, 1, 1, y_dim).unsqueeze(1).float() # batch, 1, z_dim, x_dim, y_dim
            xx_channel = xx_channel.permute(0, 1, 3, 4, 2) # batch, 1, x, y, z

            yy_ones = torch.ones([batch_size_tensor, x_dim], dtype=torch.float32).cuda()
            yy_ones = yy_ones[:, :, None]  # (batch, 175, 1)
            yy_range = self.list_K[1][i]
            yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
            yy_range = yy_range[:, None, :]
            yy_channel = torch.matmul(yy_ones, yy_range)
            yy_channel = yy_channel.unsqueeze(3).repeat(1, 1, 1, z_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
            yy_channel = yy_channel.permute(0, 1, 2, 3, 4)

            zz_ones = torch.ones([batch_size_tensor, y_dim], dtype=torch.float32).cuda()
            zz_ones = zz_ones[:, :, None]  # (batch, 175, 1)
            zz_range = self.list_K[2][i]
            zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
            zz_range = zz_range[:, None, :]
            zz_channel = torch.matmul(zz_ones, zz_range)
            zz_channel = zz_channel.unsqueeze(3).repeat(1, 1, 1, x_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
            zz_channel = zz_channel.permute(0, 1, 4, 2, 3)

            ## TODO : normalize w matrix
            large_w = (xx_channel + yy_channel + zz_channel)
            # large_w = nn.Softmax(-1)(large_w.contiguous().view(large_w.size()[0], large_w.size()[1], -1)).view_as(large_w)
            large_w = nn.Sigmoid()(large_w)
            f_out += large_w * out

        """ bias """
        xx_range = self.list_parameter_b[0]
        xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)
        xx_range = xx_range[:, None, :, None, None]

        yy_range = self.list_parameter_b[1]
        yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
        yy_range = yy_range[:, None, None, :, None]

        zz_range = self.list_parameter_b[2]
        zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
        zz_range = zz_range[:, None, None, None, :]

        ww_range = self.list_parameter_b[3] # [a]
        ww_range = ww_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, w_dim)  # (batch, 200)
        ww_range = ww_range[:, :, None, None, None]

        f_out = f_out + xx_range + yy_range + zz_range + ww_range
        f_out = self.bn(f_out)
        f_out = self.act_func(f_out)
        return f_out

class BasicConv_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, act_func='relu', bn=True, bias=False):
        super(BasicConv_Block, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes) if bn else None
        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x


def calcu_featureMap_dim(input_size, kernel, stride, padding, dilation):
    padding = np.tile(padding, len(input_size))
    kernel = np.tile(kernel, len(input_size))
    stride = np.tile(stride, len(input_size))
    dilation = np.tile(dilation, len(input_size))

    t_inputsize = np.array(input_size) + (padding * 2)
    t_kernel = (kernel-1) * dilation + 1
    output_size = (t_inputsize - t_kernel) // stride + 1
    return output_size


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()

        self.with_r = with_r

    def forward(self, input_tensor):
        # batch, 1, x, y, z
        x_dim = input_tensor.shape[2]
        y_dim = input_tensor.shape[3]
        z_dim = input_tensor.shape[4]
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([batch_size_tensor, z_dim], dtype=torch.float32).cuda()
        xx_ones = xx_ones[:, :, None]  # (batch, 175, 1)
        xx_range = torch.arange(0, x_dim, dtype=torch.float32).cuda()  # (200,)
        xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)  # (batch, 200)
        xx_range = xx_range[:, None, :]  # (batch, 1, 200)
        xx_channel = torch.matmul(xx_ones, xx_range) # (4, 175, 200)
        xx_channel = xx_channel.unsqueeze(3).repeat(1, 1, 1, y_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del xx_ones, xx_range
        xx_channel /= (x_dim - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.permute(0,1,3,4,2)

        yy_ones = torch.ones([batch_size_tensor, x_dim], dtype=torch.float32).cuda()
        yy_ones = yy_ones[:, :, None]  # (batch, 175, 1)
        yy_range = torch.arange(0, y_dim, dtype=torch.float32).cuda()  # (200,)
        yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
        yy_range = yy_range[:, None, :]  # (batch, 1, 200)
        yy_channel = torch.matmul(yy_ones, yy_range) # (4, 175, 200)
        yy_channel = yy_channel.unsqueeze(3).repeat(1, 1, 1, z_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del yy_ones, yy_range
        yy_channel /= (y_dim - 1)
        yy_channel = yy_channel * 2 - 1
        yy_channel = yy_channel.permute(0, 1, 2, 3, 4)

        zz_ones = torch.ones([batch_size_tensor, y_dim], dtype=torch.float32).cuda()
        zz_ones = zz_ones[:, :, None]  # (batch, 175, 1)
        zz_range = torch.arange(0, z_dim, dtype=torch.float32).cuda()  # (200,)
        zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
        zz_range = zz_range[:, None, :]  # (batch, 1, 200)
        zz_channel = torch.matmul(zz_ones, zz_range) # (4, 175, 200)
        zz_channel = zz_channel.unsqueeze(3).repeat(1, 1, 1, x_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del zz_ones, zz_range
        zz_channel /= (z_dim - 1)
        zz_channel = zz_channel * 2 - 1
        zz_channel = zz_channel.permute(0, 1, 4, 2, 3)
        # ret = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], 1)
        ret = torch.cat([xx_channel, yy_channel, zz_channel], 1)
        return ret

class CoordConv(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1 , groups = 1, bias=False, with_r = False):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        self.conv = nn.Conv3d(in_channels+3, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret



class sign_sqrt(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input) * torch.sqrt(torch.abs(input))
        # output = torch.sqrt(input.abs())
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        grad_input = torch.div(grad_output, ((torch.abs(output)+0.03)*2.))
        return grad_input

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class XceptionConv_layer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(XceptionConv_layer, self).__init__()
        self.out_channels = out_planes
        self.conv = SeparableConv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Multi_Heads_Self_Attn(nn.Module):
    def __init__(self, n_featuremap, n_heads = 4,  d_k = 16):
        super(Multi_Heads_Self_Attn, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k , kernel_size=1, padding=0, bias=False)

        """ key """
        self.key_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)

        """ value """
        self.value_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv3d(self.num_heads * self.d_k, n_featuremap, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(n_featuremap)
        self.relu = nn.ReLU(inplace=True)

        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, width, height, depth = x.size()
        total_key_depth = width * height * depth

        """ linear for each component"""
        queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys )
        values = self._split_heads(values)

        """ query scale"""
        query_scale = np.power(total_key_depth // self.num_heads, -0.5)
        queries *= query_scale

        """Combine queries and keys """
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        weights = self.softmax(logits)  # BX (N) X (N/p)

        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, height, depth, -1).permute(0, 4, 1, 2, 3)
        out = self.output_conv(out)
        out = self.bn(out)

        """ residual """
        out = self.gamma * out
        out = out + x

        # x = (1-self.gamma) * x
        # out = torch.cat((out, x), 1)

        out = self.relu(out)
        return out, weights, self.gamma
