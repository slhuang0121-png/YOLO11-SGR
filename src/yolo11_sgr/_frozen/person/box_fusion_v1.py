"""Shared deterministic same-category NMS for all region-selection policies.

Public interface: original-image xyxy, confidence, 1-based category ID.
No score calibration or crop-border filter; their effects need separate ablations.
"""
import numpy as np

def validate_predictions(rows, class_count=10):
    values=np.asarray(rows,dtype=np.float64).reshape(-1,6).copy()
    if not np.isfinite(values).all():raise ValueError('Nonfinite prediction')
    if len(values):
        if np.any(values[:,2:4]<=values[:,:2]):raise ValueError('Nonpositive box')
        if np.any((values[:,4]<0)|(values[:,4]>1)):raise ValueError('Invalid confidence')
        if np.any(values[:,5]!=np.floor(values[:,5])) or np.any((values[:,5]<1)|(values[:,5]>class_count)):
            raise ValueError('Invalid category mapping')
    return values

def visdrone_rows_to_xyxy(rows):
    values=np.asarray(rows,dtype=np.float64).reshape(-1,8)
    result=values[:,:6].copy()
    result[:,2:4]+=result[:,:2]
    return validate_predictions(result)

def xyxy_to_visdrone_rows(rows):
    values=validate_predictions(rows)
    result=np.empty((len(values),8),dtype=np.float64)
    result[:,:6]=values;result[:,2:4]-=result[:,:2];result[:,6:]=-1
    return result

def project_crop_predictions(rows,window,width,height,class_count=10):
    values=validate_predictions(rows,class_count)
    x1,y1,x2,y2=window
    if not(0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('Window outside original image')
    if len(values) and (np.any(values[:,:2]<-1e-5) or np.any(values[:,2]>x2-x1+1e-5) or np.any(values[:,3]>y2-y1+1e-5)):
        raise ValueError('Crop predictions must already have letterbox padding removed and be clipped')
    values[:,[0,2]]+=x1;values[:,[1,3]]+=y1
    return values

def pair_iou(box,others):
    overlap=np.maximum(0,np.minimum(box[2:4],others[:,2:4])-np.maximum(box[:2],others[:,:2])).prod(axis=1)
    area=(box[2]-box[0])*(box[3]-box[1])
    other_area=(others[:,2]-others[:,0])*(others[:,3]-others[:,1])
    return overlap/(area+other_area-overlap)

def fuse_predictions(groups,iou_threshold=.7,max_det=1000,class_count=10):
    """Groups start with global predictions, then crops in selected order.

    Ties preserve group/row order. The all-category cap is applied AFTER NMS,
    before any personnel-only metric. IoU > threshold suppresses, as native NMS.
    Provenance allows diagnostics of global boxes replaced by crop boxes.
    """
    if not(0<=iou_threshold<1) or not isinstance(max_det,int) or max_det<1:raise ValueError('Bad fusion settings')
    groups=[validate_predictions(group,class_count) for group in groups]
    values=np.concatenate(groups,axis=0) if groups else np.empty((0,6))
    origins=np.concatenate([np.full(len(group),index,dtype=int) for index,group in enumerate(groups)]) if groups else np.empty(0,dtype=int)
    order=np.argsort(-values[:,4],kind='stable')
    remaining=order.copy();kept=[];suppressed=[]
    while len(remaining):
        best=int(remaining[0]);kept.append(best)
        rest=remaining[1:]
        same=values[rest,5]==values[best,5]
        remove=np.zeros(len(rest),dtype=bool)
        if same.any():remove[same]=pair_iou(values[best,:4],values[rest[same],:4])>iou_threshold
        suppressed.extend(dict(suppressed_index=int(index),kept_index=best,
                               suppressed_group=int(origins[index]),kept_group=int(origins[best])) for index in rest[remove])
        remaining=rest[~remove]
    limited=kept[:max_det]
    diagnostics=dict(input_count=len(values),after_nms_count=len(kept),output_count=len(limited),
                     truncated_by_all_category_cap=len(kept)-len(limited),
                     kept_indices=limited,kept_groups=[int(origins[index]) for index in limited],
                     suppressed=suppressed,
                     global_suppressed_by_crop=sum(x['suppressed_group']==0 and x['kept_group']>0 for x in suppressed),
                     fusion='same-category greedy NMS, uncalibrated scores, stable global-first ties')
    return values[limited].copy(),diagnostics
