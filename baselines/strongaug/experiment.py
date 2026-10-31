"""Isolated native-training B1 repair study; no synthetic or formal test input."""
from pathlib import Path
import ast,hashlib,json,sys,random,os,math
import cv2
import numpy as np
from PIL import Image
from augment import seed_for,legacy_rng,complete,RECIPE

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def canonical(d):return hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def legacy_functions(source):
    tree=ast.parse(Path(source).read_text(encoding='utf-8-sig'))
    wanted=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('strong_aug','seg_loss')]
    assert {n.name for n in wanted}=={'strong_aug','seg_loss'}
    import torch
    import torch.nn.functional as F
    ns={'np':np,'cv2':cv2,'random':random,'torch':torch,'F':F,'lc':{'kappa_s':4.}}
    exec(compile(ast.Module(body=wanted,type_ignores=[]),str(source),'exec'),ns)
    return ns['strong_aug'],ns['seg_loss']

def read_images(rows,include_masks=True):
    out=[]
    for r in rows:
        if r['split'] not in ('train','val') or r['pool']!='train_positive':raise ValueError('native train/val only')
        assert sha(r['image_path'])==r['image_sha256'] and sha(r['mask_path'])==r['mask_sha256']
        x=np.asarray(Image.open(r['image_path']).convert('RGB'),np.uint8)
        m=np.asarray(Image.open(r['mask_path']).convert('L'),np.uint8)
        if m.max()<=1:y=m>0
        elif m.max()>=128:y=m>127
        else:raise ValueError('ambiguous mask')
        assert x.shape[:2]==y.shape==tuple(r['shape'])
        out.append((x,y))
    return out

def batch(data,rows,seed,step,legacy,batch_size=32,crop=256):
    rng=np.random.default_rng(seed_for(seed,step,'samples'))
    xs=[];ys=[];vs=[];tokens=[]
    with legacy_rng(seed_for(seed,step,'legacy_photometry')):
        for j in range(batch_size):
            idx=int(rng.integers(len(data)));im,mask=data[idx];h,w=mask.shape
            y=int(rng.integers(max(1,h-crop+1)));x=int(rng.integers(max(1,w-crop+1)));k=int(rng.integers(4));flip=bool(rng.integers(2))
            ci=np.zeros((crop,crop,3),np.uint8);cm=np.zeros((crop,crop),np.float32);valid=np.zeros_like(cm)
            hh=min(crop,h-y);ww=min(crop,w-x);ci[:hh,:ww]=im[y:y+hh,x:x+ww];cm[:hh,:ww]=mask[y:y+hh,x:x+ww];valid[:hh,:ww]=1
            arr=[np.rot90(a,k) for a in [ci,cm,valid]]
            if flip:arr=[a[:,::-1] for a in arr]
            ci,cm,valid=[np.ascontiguousarray(a) for a in arr];ci=legacy(ci)
            xs.append(ci);ys.append(cm);vs.append(valid);tokens.append([rows[idx]['image_id'],y,x,k,flip])
    image=np.stack(xs);target=np.stack(ys);valid=np.stack(vs);zero=np.zeros_like(target)
    labels=[target,zero,valid,zero.copy()]
    newx,newy,events=complete(image,labels,seed,step)
    return (image,labels),(newx,newy),{'tokens':tokens,'source_crop_sha256':canonical(tokens),'legacy_image_sha256':array_sha(image),'completed_image_sha256':array_sha(newx),'events':events}

def tensor_batch(item,device):
    import torch
    x,ys=item;mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
    x=np.ascontiguousarray(((x.astype(np.float32)/255-mean)/std).transpose(0,3,1,2))
    return [torch.from_numpy(x).to(device)]+[torch.from_numpy(np.ascontiguousarray(y[:,None])).to(device) for y in ys]

def runtime():
    import torch,segmentation_models_pytorch as smp
    return {'python':sys.version,'numpy':np.__version__,'cv2':cv2.__version__,'torch':str(torch.__version__),'smp':smp.__version__,'cuda':torch.version.cuda}

def verify_files(root):
    p=load(root/'protocol.json')
    for name,h in p['code_sha256'].items():assert sha(root/name)==h,name
    assert sha(root/'bindings.json')==p['bindings_sha256']
    assert sha(root/'preflight.json')==p['preflight_sha256']
    assert p['augmentation']==RECIPE
    b=load(root/'bindings.json')
    assert sha(b['freeze_path'])==b['freeze_sha256']
    assert sha(b['split_path'])==b['split_sha256']
    for name,h in b['protected_production'].items():assert sha(name)==h,name
    return p,b

def cpu_tree(x):
    import torch
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x

def save_state(path,nets,opts,next_step,contract,chain):
    import torch
    payload={'schema':'native_B1_matched_pair_resume_v1','contract':contract,'next_step':next_step,'schedule_total':80000,'input_chain':chain,
             'models':[cpu_tree(n.state_dict()) for n in nets],'optimizers':[cpu_tree(o.state_dict()) for o in opts]}
    tmp=Path(str(path)+'.partial')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    dump(Path(str(path)+'.receipt.json'),{'sha256':sha(path),'next_step':next_step,'contract':contract})

def restore_state(path,nets,opts,contract):
    import torch
    receipt=load(Path(str(path)+'.receipt.json'));assert sha(path)==receipt['sha256'] and receipt['contract']==contract
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['contract']==contract and state['schema']=='native_B1_matched_pair_resume_v1'
    assert state['next_step']==receipt['next_step'] and 0<=state['next_step']<=80000
    assert len(state['models'])==len(state['optimizers'])==len(nets)==len(opts)==2
    for net,opt,m,o in zip(nets,opts,state['models'],state['optimizers']):net.load_state_dict(m,strict=True);opt.load_state_dict(o)
    return state['next_step'],state['input_chain']

def forward_backward(net,opt,tensors,loss_fn,seed,step,device,update=True):
    import torch
    # Counter-based model RNG is identical across arms, and independent of augmentation RNG.
    key=seed_for(seed,step,'model_dropout') & ((1<<63)-1)
    torch.manual_seed(key)
    if device.type=='cuda':torch.cuda.manual_seed(key)
    opt.zero_grad(set_to_none=True);pred=net(tensors[0]);loss=loss_fn(pred,*tensors[1:])
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward()
    checks=[torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]
    if not checks or not bool(torch.stack(checks).all()):raise ValueError('nonfinite gradient')
    if update:opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach())
