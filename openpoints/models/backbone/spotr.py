"""Official implementation of SPoTr
Self-positioning Point-based Transformer for Point Cloud Understanding
https://arxiv.org/pdf/2303.16450
Jinyoung Park, Sanghyeok Lee, Sihyeon Kim, Yunyang Xiong, Hyunwoo J. Kim
"""
"""
Our codes are based on OpenPoints.
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
    
from torch.autograd import Variable
from einops import rearrange, repeat

def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


def get_aggregation_features(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


def new_attention(x_i, y_j=None, attn=None, aux_attn= None, tau= 1):
    if len(x_i.shape) == 3:
        attn = torch.bmm(y_j.transpose(1,2).contiguous(), x_i).detach()
        attn = nn.functional.softmax(attn, -1)#(b,m,n)
        attn = attn*aux_attn
            
        out2 = torch.bmm(x_i, attn.transpose(1,2).contiguous()) #(b,d,m)
        return out2
    else:
        b, d, n_s, n_g = x_i.shape
        channel_attn = (nn.functional.softmax(attn/tau, -1)) #(b, d, n_s n_g)
        channel_attn = channel_attn
        out1 = ((channel_attn))* x_i #(b, d, n_s, n_g) 
        if aux_attn is not None:
            out1 = out1 * aux_attn.unsqueeze(1)#(b,d,n,m) (b 1 n m) -> (b d n m)
        return out1.sum(-1)
            



class SPALPA(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """
    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 gamma=16,
                 num_gp=16,
                 tau_delta=1,
                 **kwargs
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type
        self.gamma =gamma
        self.tau_delta=tau_delta
        self.use_global = True if (not self.is_head) and (not self.all_aggr) else False
        
        self.alpha=nn.Parameter(torch.zeros((1,), dtype=torch.float32)) 
        
        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)
            
        # actually, one can use local aggregation layer to replace the following
        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_conv(channels[i], channels[i + 1],
                                     norm_args=norm_args if not is_head else None,
                                     act_args=None if i == len(channels) - 2
                                     and (self.use_res or is_head) else act_args,
                                     **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        if not self.all_aggr:
            self.attn_local = create_conv(channels[0], channels[-1],
                                        norm_args=norm_args,
                                        act_args=None,
                                        **conv_args)
        
        if self.use_global:
            gconvs= []
            for i in range(len(channels) - 1):
                gconvs.append(create_conv(channels[i], channels[i + 1],
                            norm_args=norm_args,
                            act_args=None if i == len(channels) - 2
                            and (self.use_res) else act_args,
                            **conv_args)
                )
                
                
            self.gconvs = nn.Sequential(*gconvs)
            self.attn_global = create_conv(channels[0], channels[-1],
                                        norm_args=norm_args,
                                        act_args=None,
                                        **conv_args)
            self.z = nn.Parameter(torch.randn(num_gp, in_channels))  #(m, d)
        
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
                
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

                
    def forward(self, pf):
        p, f = pf
        
        if self.is_head:
            f = self.convs(f)  # (n, c)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
                
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
                dp, fj = self.grouper(new_p, p, f)  #(b,3,n_s,k), (b, hidden_dim, n_s, k)

                B, D, N_s = fi.shape #

                fj = get_aggregation_features(new_p, dp, fi, fj, feature_type=self.feature_type)
                updated_local_f = new_attention(self.convs(fj), attn = self.attn_local(fj))
                if self.use_global:
                    z = repeat(self.z, 'm d -> b m d', b=B).contiguous()
                    interpolation_map = torch.bmm(z,fi) # (b,m,d)(b,d,n),  -> (b, m, n)
                    
                    global_p = torch.bmm((interpolation_map).softmax(-1), new_p) # (b,m,n),(b,n,3) -> (b, m, 3)
                    dist = torch.cdist(global_p, new_p) # (b,m,3),(b,n,3) -> (b,m,n)
                    g_kern = torch.exp(-self.gamma*dist.pow(2)) #(b, m, n)
                    
                    global_f = new_attention(fi, z.transpose(1,2), aux_attn=g_kern, tau=self.tau_delta)


                    global_dp = rearrange(global_p[:,None,:,:]-new_p[:,:,None,:], 'b n m d -> b d n m').contiguous() # b 3 m n
                    global_fj = repeat(global_f, 'b d m -> b d n m', n=new_p.size(1)).contiguous()
                    global_fj = get_aggregation_features(new_p, global_dp, fi, global_fj, feature_type=self.feature_type) # (b,d+3,n',m)
                    
                    updated_global_f = new_attention(self.gconvs(global_fj), attn = self.attn_global(global_fj), tau=self.tau_delta)# (b,d+3,n',m) -> (b,d',n')

                    
                    alpha = self.alpha.sigmoid()
                    f = updated_local_f*(1-alpha) + updated_global_f*alpha


            else:
                fi = None
                dp, fj = self.grouper(new_p, p, f)  #(b,3,n_s,k), (b, hidden_dim, n_s, k)
                fj = get_aggregation_features(new_p, dp, fi, fj, feature_type=self.feature_type)
                f = self.pool(self.convs(fj))

            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


# LPA + MLP block
class LPAMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 gamma=16,
                 num_gp=16,
                 tau_delta=1,
                 **kwargs
                 ):
        super().__init__()
        
        self.gamma = gamma
        self.num_gp = num_gp
        self.tau_delta = tau_delta
        
        self.use_res = use_res
        self.feature_type = aggr_args['feature_type']
        channels = [in_channels, in_channels,in_channels]
        
        channels[0] = CHANNEL_MAP[self.feature_type](channels[0])
        convs = []
        gconvs = []

        self.attn_local = create_convblock2d(channels[0], channels[-1],
                                norm_args=norm_args,
                                act_args=None,
                                **conv_args)
        
        for i in range(len(channels) - 1):
            convs.append(create_convblock2d(channels[i], channels[i + 1],
                                    norm_args=norm_args,
                                    act_args=None if i == len(channels) - 2 else act_args,
                                    **conv_args)
                        )
        
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        
        
        mid_channels = int(in_channels * expansion)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        ffn = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            ffn.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.ffn = nn.Sequential(*ffn)
        self.act = create_act(act_args)

        self.alpha=nn.Parameter(torch.zeros((1,), dtype=torch.float32)) 
        
    def forward(self, pf):
        p,f = pf
        
        identity = f
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_features(p, dp, f, fj, self.feature_type)
        
        f = new_attention(self.convs(fj), attn = self.attn_local(fj))

        f = self.act(f+identity)
        identity=f
        
        f = self.ffn(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        
        return p,f



@MODELS.register_module()
class SPoTrEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 5, 5, 5, 5],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str = 'LPAMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 num_layers: int = 1,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.num_layers = num_layers
        self.use_res = kwargs.get('use_res', True)

        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)
        
        gamma = kwargs.get('gamma', 16)
        num_gp = kwargs.get('num_gp', 16)
        tau_delta = kwargs.get('tau_delta', 1)
        
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')
        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1, 
                gamma=gamma, num_gp=num_gp, tau_delta=tau_delta,
            ))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels
        
    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False,
                  gamma=16, num_gp=8, tau_delta=1):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SPALPA(self.in_channels, channels,
                                     self.num_layers if not is_head else 1, stride,
                                     group_args=group_args, norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     sampler=self.sampler, use_res=self.use_res, is_head=is_head,
                                     gamma=gamma,
                                     num_gp=num_gp, tau_delta=tau_delta,
                                     **self.aggr_args
                                     ))
     
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res, gamma=gamma, num_gp=num_gp, tau_delta=1
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()

        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
            p.append(p0)
            f.append(f0)
        dicts = {"p":p, "f" :f}
        return f0.squeeze(-1), dicts

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        
        return p, f

    def forward(self, p0, x0=None):
        self.forward_seg_feat(p0, x0)

@MODELS.register_module()
class SPoTrDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        return f[-len(self.decoder) - 1]



@MODELS.register_module()
class SPoTrPartDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 decoder_strides: List[int] = [4, 4, 4, 4],
                 act_args: str = 'relu',
                 num_classes: int = 16,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        fp_channels = encoder_channel_list[:-1]
        
        # the following is for decoder blocks
        self.conv_args = kwargs.get('conv_args', None)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)
        block = kwargs.get('block', 'LPAMLP')
        if isinstance(block, str):
            block = eval(block)
        self.blocks = decoder_blocks
        self.strides = decoder_strides
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.expansion = kwargs.get('expansion', 4)
        radius = kwargs.get('radius', 0.1)
        nsample = kwargs.get('nsample', 16)
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        self.num_classes = num_classes
        self.use_res = kwargs.get('use_res', True)
        group_args = kwargs.get('group_args', {'NAME': 'ballquery'})
        self.aggr_args = kwargs.get('aggr_args', 
                                    {'feature_type': 'dp_fj', "reduction": 'max'}
                                    )  

        # global features
        self.global_conv2 = nn.Sequential(
            create_convblock1d(fp_channels[-1] * 2, 128,
                                norm_args=None,
                                act_args=act_args))
        self.global_conv1 = nn.Sequential(
            create_convblock1d(fp_channels[-2] * 2, 64,
                                norm_args=None,
                                act_args=act_args))
        skip_channels[0] += 64 + 128 + 16  # shape categories labels


        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i], group_args=group_args, block=block, blocks=self.blocks[i])

        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels, group_args=None, block=None, blocks=1):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, act_args=self.act_args))
        self.in_channels = fp_channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def forward(self, p, f, cls_label):
        B, N = p[0].shape[0:2]

        emb1 = self.global_conv1(f[-2])
        emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
        emb2 = self.global_conv2(f[-1])
        emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
        cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
        cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1)
        cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
        cls_one_hot = cls_one_hot.expand(-1, -1, N)

        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i-1], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        f[-len(self.decoder) - 1] = self.decoder[0][1:](
            [p[1], self.decoder[0][0]([p[1], torch.cat([cls_one_hot, f[1]], 1)], [p[2], f[2]])])[1]

        return f[-len(self.decoder) - 1]
