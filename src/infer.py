import torch
from torchvision.transforms.functional import gaussian_blur

def post_process(maps):
    # Reshape B*C into a single leading dimension for torchvision version compatibility.
    b, c, h, w = maps.shape
    qcs = gaussian_blur(maps[:,:3].float().reshape(b*3,h,w), [9,9], [2.,2.]).reshape(b,3,h,w)
    width = gaussian_blur(maps[:,3].float(), [5,5], [1.,1.])[:,None]
    return torch.cat([qcs,width],dim=1)

def select_peak(q, tol=1e-4):
    """Pixel index of the top-1 grasp in each smoothed quality map q (B,H,W).

    A plain argmax is ill-defined on a flat plateau (e.g. a ground-truth Q block): torch returns the first
    (top-left) maximum, which can lie tens of pixels from the rectangle centre. Instead, all pixels within
    tol of the maximum form a plateau mask, and the selected pixel is the most interior one of that mask
    (maximum of the Gaussian-blurred mask, restricted to the mask). For a single strict maximum this
    reduces to argmax.
    """
    b, h, w = q.shape
    peak = q.reshape(b, -1).amax(dim=1)[:, None, None]
    mask = (q >= peak - tol).float()
    depth = gaussian_blur(mask[:, None], [31, 31], [8., 8.])[:, 0] * mask
    return depth.reshape(b, -1).argmax(dim=1)

def decode_top1(maps, cfg):
    smooth = post_process(maps)
    b, _, h, w = smooth.shape
    if getattr(cfg, 'decode', 'plateau') == 'argmax':  # decoder used for the runs before the plateau fix
        indices = smooth[:,0].reshape(b,-1).argmax(dim=1)
    else:
        indices = select_peak(smooth[:,0])
    r, c = indices//w, indices%w
    batch = torch.arange(b,device=maps.device)
    cosine, sine = smooth[batch,1,r,c], smooth[batch,2,r,c]
    angle = .5*torch.atan2(sine,cosine)
    length = (smooth[batch,3,r,c]*cfg.width_norm).clamp(min=1.0)
    scale = cfg.img/416.0
    return torch.stack([c.float()/scale,r.float()/scale,length/scale,length/(2*scale),angle],dim=1)
