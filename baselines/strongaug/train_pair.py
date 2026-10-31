from pathlib import Path
import argparse,os,sys,time,json,traceback
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(Path(__file__).resolve().parent)]
import torch,segmentation_models_pytorch as smp
from experiment import *

def atomic_json(path,value):
    tmp=Path(str(path)+'.partial');dump(tmp,value);os.replace(tmp,path)

@torch.no_grad()
def validate(net,values,device):
    net.eval();tp=fp=fn=0
    mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
    for image,mask in values:
        h,w=mask.shape;H,W=max(32,(h+31)//32*32),max(32,(w+31)//32*32)
        x=(image.astype(np.float32)/255-mean)/std
        x=np.pad(x,((0,H-h),(0,W-w),(0,0)),mode='reflect')
        pred=(torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1)[None])).to(device)))[0,0,:h,:w].cpu().numpy()>=.5)
        tp+=int((pred&mask).sum());fp+=int((pred&~mask).sum());fn+=int((~pred&mask).sum())
    net.train()
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)}

def main(seed,resume):
    protocol,bindings=verify_files(ROOT);pre=load(ROOT/'preflight.json')
    assert runtime()==pre['runtime'];assert seed in [0,1,2]
    out=Path(protocol['output_directory'])/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('already complete; no overwrite')
    if (out/'progress.json').exists() and not resume:raise ValueError('existing progress requires explicit --resume')
    init=next(r for r in pre['initial'] if r['seed']==seed);assert sha(init['path'])==init['sha256']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Preserve original adaptive pooling implementation. Backward is not bitwise-deterministic on CUDA;
    # this is disclosed, and exact resumed trajectory is NOT claimed.
    torch.use_deterministic_algorithms(False)
    device=torch.device('cuda:0');nets=[];opts=[]
    initial_state=torch.load(init['path'],map_location='cpu',weights_only=True)
    for _ in range(2):
        net=smp.UPerNet('tu-convnext_tiny',encoder_weights=None,classes=1)
        net.load_state_dict(initial_state,strict=True);net.to(device).train();nets.append(net)
        opts.append(torch.optim.AdamW(net.parameters(),lr=6e-5,weight_decay=.01))
    assert all(torch.equal(nets[0].state_dict()[k],nets[1].state_dict()[k]) for k in initial_state)
    del initial_state
    contract={'scope':'registered_posthoc_native_B1_coverage_repair_only','protocol_sha256':sha(ROOT/'protocol.json'),'seed':seed,'initial_sha256':init['sha256'],'runtime':runtime(),'total_steps':80000,'batch':32,'crop':256,'arms':['B1_legacy_matched','B1_completed_matched'],'resume_bitwise_cuda_claim':False}
    next_step=0;chain='0'*64
    if resume:
        pointer=load(out/'latest_checkpoint.json');cp=out/pointer['filename']
        assert sha(cp)==pointer['sha256'];next_step,chain=restore_state(cp,nets,opts,contract)
    train=[r for r in bindings['train']];vals=[r for r in bindings['val']]
    data=read_images(train);validation=read_images(vals);legacy,loss_fn=legacy_functions(ROOT/'source/legacy_student_v1.py')
    start=time.monotonic();total=80000
    atomic_json(out/'progress.json',{'status':'running','seed':seed,'next_step':next_step,'total_steps':total,'pid':os.getpid(),'protocol_sha256':contract['protocol_sha256']})
    attempt=out/f'attempt_{time.time_ns()}';attempt.mkdir()
    dump(attempt/'start.json',{'resume':resume,'next_step':next_step,'contract':contract})
    log=(attempt/'training.jsonl').open('a',encoding='utf-8',buffering=1)
    for step in range(next_step,total):
        a,b,trace=batch(data,train,seed,step,legacy)
        # Both arms consume exactly this source/crop multiset. Completed branch then changes augmentation.
        chain=hashlib.sha256((chain+trace['source_crop_sha256']).encode()).hexdigest()
        losses=[]
        for net,opt,item in zip(nets,opts,[a,b]):
            for group in opt.param_groups:group['lr']=6e-5*(1-step/total)
            losses.append(forward_backward(net,opt,tensor_batch(item,device),loss_fn,seed,step,device))
        if step==0 or (step+1)%1000==0:
            entry={'next_step':step+1,'loss_legacy':losses[0],'loss_completed':losses[1],'elapsed_since_start_s':time.monotonic()-start,'source_crop_sha256':trace['source_crop_sha256'],'cumulative_source_chain':chain,'lr_used':opts[0].param_groups[0]['lr']}
            log.write(json.dumps(entry)+'\n');atomic_json(out/'progress.json',dict(entry,status='running',seed=seed,total_steps=total,pid=os.getpid(),protocol_sha256=contract['protocol_sha256']))
        if (step+1)%5000==0:
            for arm,net in zip(contract['arms'],nets):
                entry={'step':step+1,'arm':arm,'native_val_fixed05':validate(net,validation,device),'checkpoint_selection':'fixed80000only'}
                with (attempt/'validation.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
        if (step+1)%10000==0:
            cp=out/f'checkpoint_{step+1:06d}.pt';assert not cp.exists()
            save_state(cp,nets,opts,step+1,contract,chain)
            atomic_json(out/'latest_checkpoint.json',{'filename':cp.name,'sha256':sha(cp),'next_step':step+1})
    files=[]
    for arm,net in zip(contract['arms'],nets):
        p=out/(arm+'_final.pt');assert not p.exists();torch.save(cpu_tree(net.state_dict()),p);files.append({'arm':arm,'path':str(p),'sha256':sha(p)})
    verify_files(ROOT)
    final={'status':'completed','seed':seed,'completed_steps_per_arm':80000,'arms':files,'contract':contract,'source_chain':chain,'training_seconds':time.monotonic()-start,'formal_test_images':0,'scientific_H1_H2_decision':False}
    dump(out/'complete.json',final);atomic_json(out/'progress.json',final);log.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    try:main(a.seed,a.resume)
    except BaseException as e:
        try:
            protocol=load(ROOT/'protocol.json');out=Path(protocol['output_directory'])/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
            dump(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'seed':a.seed,'status':'failed','automatic_retry':False})
        finally:raise
