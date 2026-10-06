"""Supplemental person FROC; not the official VisDrone/COCO AP metric.

Frames have all-category original xyxy predictions (Nx6), evaluable personnel
boxes, focus mask, and genuine ignored-person boxes. Category-0 region filtering
is performed by the separately verified official-toolkit adapter BEFORE calling
this module. Nonfocus people are singly matched ignores; genuine ignored people
accept repeated detections by intersection / detection area. Other object classes
are never used as ignores for a person prediction.
"""
import numpy as np
from box_fusion_v1 import validate_predictions,pair_iou

def boxes(values):
    result=np.asarray(values,dtype=np.float64).reshape(-1,4)
    if not np.isfinite(result).all() or np.any(result[:,2:4]<=result[:,:2]):raise ValueError('Bad GT boxes')
    return result

def match_frame(predictions,person_boxes,focus_mask=None,ignored_person_boxes=(),
                person_ids=(1,2),class_count=10,iou=.5,ignore_iof=.5,max_det=1000):
    if not 0<iou<=1 or not 0<ignore_iof<=1:raise ValueError('Invalid overlap threshold')
    if not isinstance(max_det,int) or max_det<1:raise ValueError('Invalid cap')
    values=validate_predictions(predictions,class_count)
    values=values[np.argsort(-values[:,4],kind='stable')][:max_det]
    values=values[np.isin(values[:,5],person_ids)]
    gt=boxes(person_boxes);ignored=boxes(ignored_person_boxes)
    focus=np.ones(len(gt),dtype=bool) if focus_mask is None else np.asarray(focus_mask,dtype=bool)
    if focus.shape!=(len(gt),):raise ValueError('Focus mask must cover all evaluable people')
    matched=np.zeros(len(gt),dtype=bool);events=[]
    for prediction in values:
        overlaps=pair_iou(prediction[:4],gt) if len(gt) else np.empty(0)
        candidates=np.flatnonzero(focus&~matched&(overlaps>=iou))
        if len(candidates):
            chosen=int(candidates[np.argmax(overlaps[candidates])]);matched[chosen]=True
            events.append(dict(score=float(prediction[4]),tp=1,fp=0,match='focus',gt_index=chosen))
            continue
        candidates=np.flatnonzero(~focus&~matched&(overlaps>=iou))
        if len(candidates):
            chosen=int(candidates[np.argmax(overlaps[candidates])]);matched[chosen]=True
            events.append(dict(score=float(prediction[4]),tp=0,fp=0,match='nonfocus_single_ignore',gt_index=chosen))
            continue
        detection_area=np.prod(prediction[2:4]-prediction[:2])
        intersection=np.maximum(0,np.minimum(prediction[2:4],ignored[:,2:4])-np.maximum(prediction[:2],ignored[:,:2])).prod(axis=1)
        if len(ignored) and np.any(intersection/detection_area>=ignore_iof):
            events.append(dict(score=float(prediction[4]),tp=0,fp=0,match='genuine_ignore_repeated',gt_index=None))
        else:events.append(dict(score=float(prediction[4]),tp=0,fp=1,match='false_positive',gt_index=None))
    return dict(events=events,target_count=int(focus.sum()),evaluable_person_count=len(gt),
                no_evaluable_person=len(gt)==0,ignored_person_count=len(ignored))

def curve(frames,fpi_points=(.1,.5,1.),operating_threshold=None):
    """Only whole equal-score groups produce operating points; no tie splitting.

    Reference FPI points use the attained step with greatest recall under the FP
    budget, without interpolating fractional TP/FP. They summarize this set's
    curve; a deployable threshold must be chosen on separate development data.
    """
    if not frames:raise ValueError('Empty evaluation set')
    total_targets=sum(frame['target_count'] for frame in frames)
    no_person_images=sum(frame['no_evaluable_person'] for frame in frames)
    events=[]
    for image_index,frame in enumerate(frames):
        for event in frame['events']:
            events.append(event|dict(image_index=image_index,no_evaluable_person=frame['no_evaluable_person']))
    events.sort(key=lambda event:-event['score'])
    points=[dict(threshold=None,tp=0,fp=0,recall=0. if total_targets else None,fpi=0.,
                 fp_on_no_evaluable_person_images=0)]
    tp=fp=fp_empty=0;index=0
    while index<len(events):
        threshold=events[index]['score'];end=index
        while end<len(events) and events[end]['score']==threshold:
            event=events[end];tp+=event['tp'];fp+=event['fp']
            fp_empty+=event['fp']*event['no_evaluable_person'];end+=1
        points.append(dict(threshold=threshold,tp=tp,fp=fp,recall=tp/total_targets if total_targets else None,
                           fpi=fp/len(frames),fp_on_no_evaluable_person_images=fp_empty))
        index=end
    references={}
    for target in fpi_points:
        if target<0:raise ValueError('Negative FPI budget')
        eligible=[point for point in points if point['fpi']<=target]
        best=max(eligible,key=lambda point:point['tp']) # first point means highest threshold at same recall
        references[str(target)]=best.copy()
    operating=None
    if operating_threshold is not None:
        if not np.isfinite(operating_threshold) or not 0<=operating_threshold<=1:raise ValueError('Invalid frozen threshold')
        selected=[event for event in events if event['score']>=operating_threshold]
        op_tp=sum(x['tp'] for x in selected);op_fp=sum(x['fp'] for x in selected)
        op_empty=sum(x['fp']*x['no_evaluable_person'] for x in selected)
        operating=dict(threshold=float(operating_threshold),tp=op_tp,fp=op_fp,
                       recall=op_tp/total_targets if total_targets else None,fpi=op_fp/len(frames),
                       fp_on_no_evaluable_person_images=op_empty,
                       mean_fp_per_no_evaluable_person_image=op_empty/no_person_images if no_person_images else None)
    return dict(images=len(frames),target_count=total_targets,no_evaluable_person_images=no_person_images,
                curve=points,reference_fpi_steps=references,frozen_threshold_operating_point=operating,
                ties_grouped=True,reference_points_are_not_independently_frozen_thresholds=True,
                category_0_region_filter_required=True,metric='supplemental person IoU-0.5 FROC')
