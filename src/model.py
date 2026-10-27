import math
import torch
from torch import nn
import torch.nn.functional as F

def conv(cin, cout, kernel, stride=1, padding=0, transpose=False):
    layer = nn.ConvTranspose2d if transpose else nn.Conv2d
    return nn.Sequential(layer(cin, cout, kernel, stride, padding), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels))
    def forward(self, features): return features+self.layers(features)

class FiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = nn.Linear(512, 256)
        nn.init.zeros_(self.affine.weight); nn.init.zeros_(self.affine.bias)
    def forward(self, features, text):
        with torch.autocast(device_type=features.device.type, enabled=False):
            dgamma, beta = self.affine(text.float()).chunk(2, dim=-1)
        return features*(1+dgamma.to(features.dtype)[:, :, None, None])+beta.to(features.dtype)[:, :, None, None]

class LanguageGate(nn.Module):
    """S = cos(Wp P, Wt t). With text_free=True (variant E, 'C0'), the query Wt t is replaced by one learned,
    prompt-independent vector q, so the gate still sees CLIP patch features but cannot depend on the instruction."""
    def __init__(self, text_free=False):
        super().__init__()
        self.text_free = text_free
        self.Wp = nn.Linear(768, 256)
        if text_free: self.query = nn.Parameter(torch.randn(256)/16.0)
        else: self.Wt = nn.Linear(512, 256)
        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))
    def forward(self, features, text, tokens):
        if tokens is None or tokens.shape[1:] != (196, 768):
            raise ValueError('C/D require cached patch tokens (B,196,768)')
        with torch.autocast(device_type=features.device.type, enabled=False):
            patches = F.normalize(self.Wp(tokens.float().reshape(-1, 14, 14, 768)), dim=-1)
            if self.text_free: language = F.normalize(self.query.float(), dim=-1)[None, :].expand(patches.shape[0], -1)
            else: language = F.normalize(self.Wt(text.float()), dim=-1)
            similarity = (patches*language[:, None, None, :]).sum(-1)
            logits = self.log_tau.exp()*similarity[:, None, :, :]
            attention = F.interpolate(logits.sigmoid(), size=(56, 56), mode='bilinear', align_corners=False)
        return features*(1+attention.to(features.dtype)), logits

class GraspNet(nn.Module):
    def __init__(self, variant='A'):
        super().__init__()
        if variant not in ('A','B','C','D','E'): raise ValueError(variant)
        self.variant = variant
        self.conv1, self.conv2, self.conv3 = conv(3,32,9,padding=4), conv(32,64,4,2,1), conv(64,128,4,2,1)
        self.residuals = nn.ModuleList([ResidualBlock() for _ in range(5)])
        self.up1, self.up2, self.final = conv(128,64,4,2,1,True), conv(64,32,4,2,1,True), conv(32,32,9,padding=4)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({key:nn.Conv2d(32,1,1) for key in ('q','cos','sin','width')})
        # Shared backbone initializes before any variant-specific layers.
        self.films = nn.ModuleList([FiLM() for _ in range(5)]) if variant != 'A' else nn.ModuleList()
        self.gate = LanguageGate(text_free=(variant=='E')) if variant in ('C','D','E') else None
        self.register_buffer('mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self, image, text=None, tokens=None, return_shapes=False):
        shapes = {}
        x = (image.float()/255.0-self.mean)/self.std
        for name in ('conv1','conv2','conv3'):
            x = getattr(self,name)(x)
            if return_shapes: shapes[name] = tuple(x.shape)
        t = F.normalize(text.float(), dim=-1) if self.variant != 'A' else None
        logits = None
        if self.gate is not None: x, logits = self.gate(x,t,tokens)
        if return_shapes: shapes['pre_residual'] = tuple(x.shape)
        for i, block in enumerate(self.residuals):
            x = block(x)
            if return_shapes: shapes[f'residual_{i}'] = tuple(x.shape)
            if self.films:
                x = self.films[i](x,t)
                if return_shapes: shapes[f'film_{i}'] = tuple(x.shape)
        for name in ('up1','up2','final'):
            x = getattr(self,name)(x)
            if return_shapes: shapes[name] = tuple(x.shape)
        x = self.dropout(x)
        if return_shapes: shapes['dropout'] = tuple(x.shape)
        results = []
        for name,head in self.heads.items():
            result = head(x); results.append(result)
            if return_shapes: shapes['head_'+name] = tuple(result.shape)
        maps = torch.cat(results, dim=1)
        if return_shapes: shapes['maps'] = tuple(maps.shape)
        return {'maps':maps, 'gate_logits':logits, 'shapes':shapes}

def loss_components(output, targets, cfg):
    losses = {name:F.smooth_l1_loss(output['maps'][:,i],targets[:,i])
              for i,name in enumerate(('q','cos','sin','width'))}
    gate = output['maps'].new_zeros(())
    if cfg.variant == 'D':
        target14 = (F.adaptive_max_pool2d(targets[:,0:1], (14,14)) > 0).float()
        gate = F.binary_cross_entropy_with_logits(output['gate_logits'], target14)
    losses['gate'] = gate
    losses['total'] = sum(losses[name] for name in ('q','cos','sin','width'))+cfg.lambda_gate*gate
    return losses

def parameter_counts(model):
    return {'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad)}
