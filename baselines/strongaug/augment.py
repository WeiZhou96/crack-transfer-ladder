"""Generic supervised augmentation; NOT a physically calibrated confuser operator."""
import hashlib,random,contextlib
import cv2
import numpy as np

RECIPE={'gamma':{'p':.5,'range':[.7,1.5]},'clahe':{'p':.25,'clip_limit':2.,'tile_grid':[8,8]},'motion':{'p':.25,'sizes':[3,5,7],'directions':[0,1,2,3]},'shadow':{'p':.5,'factor':[.55,.85],'vertices':[3,6],'radius_fraction':[.2,.65],'blur_kernel':5},'cutmix':{'p_batch':.5,'beta_alpha':1.,'donor':'cyclic nonzero shift of original batch; no in-place donor contamination'}}

def seed_for(seed,step,stream):
    return int.from_bytes(hashlib.sha256(f'B1-completion-v1|{seed}|{step}|{stream}'.encode()).digest()[:8],'little')

@contextlib.contextmanager
def legacy_rng(seed):
    """Keep legacy Python/MT19937 augmentation isolated from sampling/new operators."""
    ps=random.getstate();ns=np.random.get_state()
    try:
        random.seed(seed);np.random.seed(seed & 0xffffffff);yield
    finally:random.setstate(ps);np.random.set_state(ns)

def image_ops(image,rng,force=None):
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('uint8 RGB required')
    x=image.copy();h,w=x.shape[:2];events={}
    enabled=lambda k: bool(force[k]) if force is not None and k in force else rng.random()<RECIPE[k]['p']
    if enabled('gamma'):
        gamma=float(rng.uniform(*RECIPE['gamma']['range']));lut=np.rint((np.arange(256)/255.)**gamma*255).astype(np.uint8)
        x=lut[x];events['gamma']=gamma
    if enabled('clahe'):
        lab=cv2.cvtColor(x,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0]);x=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB);events['clahe']=True
    if enabled('motion'):
        k=int(rng.choice(RECIPE['motion']['sizes']));direction=int(rng.integers(4));kernel=np.zeros((k,k),np.float32)
        if direction==0:kernel[k//2,:]=1
        elif direction==1:kernel[:,k//2]=1
        elif direction==2:np.fill_diagonal(kernel,1)
        else:kernel[np.arange(k),np.arange(k-1,-1,-1)]=1
        kernel/=kernel.sum();x=cv2.filter2D(x,-1,kernel,borderType=cv2.BORDER_REFLECT_101);events['motion']={'size':k,'direction':direction}
    if enabled('shadow'):
        n=int(rng.integers(3,7));angles=np.sort(rng.uniform(0,2*np.pi,n));radius=rng.uniform(.2,.65,n)*min(h,w)
        cx,cy=rng.uniform(0,w),rng.uniform(0,h)
        points=np.rint(np.stack([cx+radius*np.cos(angles),cy+radius*np.sin(angles)],axis=1)).astype(np.int32)
        hull=cv2.convexHull(points);mask=np.zeros((h,w),np.float32);cv2.fillConvexPoly(mask,hull,1.)
        mask=cv2.GaussianBlur(mask,(5,5),0,borderType=cv2.BORDER_REFLECT_101)
        factor=float(rng.uniform(.55,.85));x=np.rint(x.astype(np.float32)*(1.-(1.-factor)*mask[:,:,None])).clip(0,255).astype(np.uint8)
        events['shadow']={'factor':factor,'points':points.tolist(),'support_pixels':int((mask>0).sum())}
    return x,events

def validate_batch(images,labels):
    if images.dtype!=np.uint8 or images.ndim!=4 or images.shape[-1]!=3 or images.shape[0]<2:raise ValueError('NHWC RGB batch>=2 required')
    if len(labels)!=4:raise ValueError('target, ignore, valid, omega fields required')
    for a in labels:
        if a.shape!=images.shape[:3] or not np.isfinite(a).all() or not np.all((a==0)|(a==1)):raise ValueError('aligned binary labels required')

def cutmix(images,labels,rng,force=None):
    validate_batch(images,labels);n,h,w,_=images.shape
    yes=(rng.random()<RECIPE['cutmix']['p_batch']) if force is None else bool(force)
    out=images.copy();labs=[a.copy() for a in labels];events=[]
    if not yes:return out,labs,events
    shift=int(rng.integers(1,n));donors=(np.arange(n)+shift)%n
    for j,d in enumerate(donors):
        lam=float(rng.beta(1,1));side=np.sqrt(1-lam);bw=max(1,int(w*side));bh=max(1,int(h*side))
        cx,cy=int(rng.integers(w)),int(rng.integers(h));x0=max(0,cx-bw//2);y0=max(0,cy-bh//2);x1=min(w,x0+bw);y1=min(h,y0+bh)
        out[j,y0:y1,x0:x1]=images[d,y0:y1,x0:x1]
        for dst,src in zip(labs,labels):dst[j,y0:y1,x0:x1]=src[d,y0:y1,x0:x1]
        events.append({'recipient':j,'donor':int(d),'box':[x0,y0,x1,y1],'retained_fraction':1-(x1-x0)*(y1-y0)/(h*w)})
    return out,labs,events

def complete(images,labels,seed,step,force=None):
    validate_batch(images,labels)
    rng=np.random.default_rng(seed_for(seed,step,'extra_photometry'))
    transformed=[];events=[]
    for image in images:
        x,e=image_ops(image,rng,force);transformed.append(x);events.append(e)
    mixrng=np.random.default_rng(seed_for(seed,step,'cutmix'))
    x,ys,mix=cutmix(np.stack(transformed),labels,mixrng,None if force is None else force.get('cutmix'))
    return x,ys,{'image_events':events,'cutmix_events':mix,'recipe':RECIPE}
