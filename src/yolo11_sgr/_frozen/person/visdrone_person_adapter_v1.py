"""Supplemental personnel evaluation with the frozen VisDrone ignore adapter."""
import numpy as np
from box_fusion_v1 import visdrone_rows_to_xyxy
from person_froc_v1 import match_frame

def prepare_frame(raw_gt,raw_dt,height,width,drop_ignored_regions,focus='all',max_det=1000):
    """drop_ignored_regions must be the independently verified official port."""
    gt=np.asarray(raw_gt,dtype=np.float64).reshape(-1,8)
    dt=np.asarray(raw_dt,dtype=np.float64).reshape(-1,8)
    if not np.isfinite(gt).all():raise ValueError('Nonfinite annotation')
    visdrone_rows_to_xyxy(dt) # validates category/score/box before filtering
    dt=dt[np.argsort(-dt[:,4],kind='stable')][:max_det]
    gt,dt=drop_ignored_regions(gt,dt,height,width)
    positive=gt[(gt[:,4]>0)&np.isin(gt[:,5],[1,2])]
    ignored=gt[(gt[:,4]==0)&np.isin(gt[:,5],[1,2])]
    boxes=positive[:,:4].copy();boxes[:,2:4]+=boxes[:,:2]
    ignored_boxes=ignored[:,:4].copy();ignored_boxes[:,2:4]+=ignored_boxes[:,:2]
    size=np.sqrt(positive[:,2]*positive[:,3])
    centers=(boxes[:,:2]+boxes[:,2:4])/2
    if len(boxes)>1:
        distances=np.sqrt(((centers[:,None,:]-centers[None,:,:])**2).sum(axis=2))
        np.fill_diagonal(distances,np.inf);nearest=distances.min(axis=1)
    else:nearest=np.full(len(boxes),np.inf)
    masks={'all':np.ones(len(boxes),dtype=bool),'raw_lt16':size<16,'raw_lt32':size<32,
           'isolated_lt32':(size<32)&(nearest>=4*size),
           'occlusion_0':positive[:,7]==0,'occlusion_1':positive[:,7]==1,'occlusion_2':positive[:,7]==2}
    if focus not in masks:raise ValueError('Unknown prespecified person stratum')
    frame=match_frame(visdrone_rows_to_xyxy(dt),boxes,masks[focus],ignored_boxes,max_det=max_det)
    frame['metadata']=dict(focus=focus,person_ids=[1,2],raw_gt_person_count=len(boxes),
                           ignored_person_count=len(ignored_boxes),all_category_detections_after_region_filter=len(dt),
                           size_definition='sqrt(raw pixel bbox width * height), no resize or clipping',
                           isolation_definition='nearest OTHER evaluable personnel center distance >= 4 * own sqrt(raw area)',
                           ignore_region_definition='frozen author-toolkit pixel-raster filter, coverage >= 0.5 removed')
    return frame
