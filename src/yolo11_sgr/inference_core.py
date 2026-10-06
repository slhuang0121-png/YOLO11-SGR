"""Verbatim frozen numerical inference functions; bootstrap supplies the paper checkpoint classes."""
from .bootstrap import activate
activate()
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.nn.modules.head import Detect

def prepare_image(image,size):
    transform=LetterBox(new_shape=(size,size),auto=False,scale_fill=False,
        scaleup=True,center=True)
    params=transform.get_params({'img':image})
    transformed=transform(image=image)
    assert transformed.shape[:2]==(size,size)
    tensor=torch.from_numpy(np.ascontiguousarray(transformed[:,:,::-1].transpose(2,0,1)))
    return tensor.unsqueeze(0).float()/255.,params

def restore_boxes(boxes,params):
    boxes=boxes.clone().float()
    boxes[:,[0,2]]=(boxes[:,[0,2]]-params['left'])/params['ratio'][0]
    boxes[:,[1,3]]=(boxes[:,[1,3]]-params['top'])/params['ratio'][1]
    h,w=params['orig_shape']
    boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,w)
    boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,h)
    return boxes

def load_model(checkpoint,mode,device,max_det,fuse):
    model=YOLO(str(checkpoint)).model.float().eval().to(device)
    heads=[module for module in model.modules() if isinstance(module,Detect)]
    assert len(heads)==1
    head=heads[0]
    before=sum(p.numel() for p in model.parameters())
    if mode=='end2end':
        if getattr(head,'one2one_cv2',None) is None:
            raise ValueError('Checkpoint has no one-to-one head')
        head.end2end=True
    elif mode=='one2many':
        if head.cv2 is None:raise ValueError('One-to-many branch already removed')
        head.end2end=False
    head.max_det=max_det
    head.xyxy=False # conventional heads feed native xywh NMS
    selected=head.end2end
    # Choose the branch BEFORE fusion; otherwise the alternative can be removed.
    if fuse:model.fuse(verbose=False)
    assert head.end2end==selected
    return model,head,{'mode':'end2end' if selected else 'one2many',
        'training_checkpoint_parameters':before,
        'inference_parameters':sum(p.numel() for p in model.parameters()),'fused':fuse}

def prediction_rows(model,head,tensor,params,conf,iou,max_det,multi_label):
    output=model(tensor)
    decoded=output[0] if isinstance(output,(list,tuple)) else output
    assert torch.isfinite(decoded).all(),'Non-finite model predictions'
    # A large explicit bound prevents timing cutoffs from silently dropping images.
    detections=non_max_suppression(output,conf_thres=conf,iou_thres=iou,
        nc=head.nc,max_det=max_det,multi_label=multi_label,end2end=head.end2end,
        max_time_img=600,max_nms=30000)[0]
    assert torch.isfinite(detections).all()
    detections=detections.detach().cpu().float()
    detections[:,:4]=restore_boxes(detections[:,:4],params)
    detections=detections[(detections[:,2]>detections[:,0])&(detections[:,3]>detections[:,1])]
    array=detections.numpy()
    array=array[np.argsort(-array[:,4],kind='stable')]
    rows=np.zeros((len(array),8),dtype=np.float64)
    rows[:,:2]=array[:,:2];rows[:,2:4]=array[:,2:4]-array[:,:2]
    rows[:,4]=array[:,4];rows[:,5]=array[:,5]+1;rows[:,6:]=-1
    return rows
