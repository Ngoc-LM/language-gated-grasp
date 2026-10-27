import torch
from torchvision.transforms.functional import gaussian_blur

def post_process(maps):
    # Reshape B*C into a single leading dimension for torchvision version compatibility.
    b, c, h, w = maps.shape
    qcs = gaussian_blur(maps[:,:3].float().reshape(b*3,h,w), [9,9], [2.,2.]).reshape(b,3,h,w)
    width = gaussian_blur(maps[:,3].float(), [5,5], [1.,1.])[:,None]
    return torch.cat([qcs,width],dim=1)

def decode_top1(maps, cfg):
    smooth = post_process(maps)
    b, _, h, w = smooth.shape
    indices = smooth[:,0].reshape(b,-1).argmax(dim=1)
    r, c = indices//w, indices%w
    batch = torch.arange(b,device=maps.device)
    cosine, sine = smooth[batch,1,r,c], smooth[batch,2,r,c]
    angle = .5*torch.atan2(sine,cosine)
    length = (smooth[batch,3,r,c]*cfg.width_norm).clamp(min=1.0)
    scale = cfg.img/416.0
    return torch.stack([c.float()/scale,r.float()/scale,length/scale,length/(2*scale),angle],dim=1)
