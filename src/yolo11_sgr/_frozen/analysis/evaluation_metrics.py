"""Explicit COCO metrics and VisDrone strata; never reuse training-validation AP.

Non-focus GT are retained as single-match ignored boxes. Score-zero GT use
crowd/IoF matching; category-zero regions use the published pixel filter.
"""
import copy
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CORE = Path(__file__).resolve().parent.parent / 'lab_modern'
if not CORE.exists():
    CORE = Path(__file__).resolve().parent.parent / '现代实验代码'
sys.path.insert(0, str(CORE))
from visdrone_toolkit_port import read_rows, drop_ignored_regions, evaluate


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


class FocusCOCOeval(COCOeval):
    """Restore explicit ignore after upstream _prepare overwrites it with iscrowd.

    Normal non-focus objects remain non-crowd, preserving IoU and one match.
    Only true score-zero annotations use crowd/IoF and repeated matching.
    """
    def _prepare(self):
        super()._prepare()
        for values in self._gts.values():
            for gt in values:
                gt['ignore']=int(bool(gt.get('iscrowd',0) or gt.get('focus_ignore',0)))


def make_coco(images, annotations, categories, detections):
    gt=COCO()
    gt.dataset={'info':{},'images':copy.deepcopy(images),
        'annotations':copy.deepcopy(annotations),'categories':copy.deepcopy(categories)}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        if detections:
            dt=gt.loadRes(copy.deepcopy(detections))
        else:
            dt=COCO()
            dt.dataset={'info':{},'images':copy.deepcopy(images),'annotations':[],
                'categories':copy.deepcopy(categories)}
            dt.createIndex()
    return gt,dt


def coco_metrics(images, annotations, categories, detections, cap=1000,
                 focus=None, image_ids=None):
    """101-recall-point, equal-category COCO AP; explicit global/per-class cap.

    area-boundary behavior in standard COCO bins is retained. Exclusive extra
    bins use nextafter to avoid counting exactly 16/32/96-pixel objects twice.
    """
    annotations=copy.deepcopy(annotations)
    for gt in annotations:
        gt['focus_ignore']=int(focus is not None and not focus(gt))
    image_ids=set(image_ids) if image_ids is not None else {i['id'] for i in images}
    grouped={i:[] for i in image_ids}
    for dt in detections:
        if dt['image_id'] in grouped:
            grouped[dt['image_id']].append(dt)
    # The global limit is applied before per-class COCO matching, in score order.
    limited=[]
    for values in grouped.values():
        limited.extend(sorted(values,key=lambda row:-row['score'])[:cap])
    gt,dt=make_coco(images,annotations,categories,limited)
    evaluator=FocusCOCOeval(gt,dt,'bbox')
    evaluator.params.imgIds=sorted(image_ids)
    evaluator.params.catIds=[c['id'] for c in categories]
    evaluator.params.maxDets=[1,10,cap]
    edges=[256.,1024.,9216.]
    evaluator.params.areaRng=[[0,float('inf')],[0,1024],[1024,9216],[9216,float('inf')],
        [0,np.nextafter(edges[0],-np.inf)],
        [edges[0],np.nextafter(edges[1],-np.inf)],
        [edges[1],np.nextafter(edges[2],-np.inf)],[edges[2],float('inf')]]
    evaluator.params.areaRngLbl=['all','small','medium','large','lt16','16to32','32to96','ge96']
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
    precision=evaluator.eval['precision'] # T,R,K,A,M
    recall=evaluator.eval['recall']       # T,K,A,M
    mean_valid=lambda a:float(a[a>-1].mean()) if np.any(a>-1) else None
    output={}
    for ai,label in enumerate(evaluator.params.areaRngLbl):
        lower,upper=evaluator.params.areaRng[ai]
        output[label]={'AP':mean_valid(precision[:,:,:,ai,-1]),
            'AP50':mean_valid(precision[0,:,:,ai,-1]),
            'AP75':mean_valid(precision[5,:,:,ai,-1]),
            'AR':mean_valid(recall[:,:,ai,-1]),
            'valid_categories':int(np.any(precision[:,:,:,ai,-1]>-1,axis=(0,1)).sum()),
            'valid_gt':int(sum(a['image_id'] in image_ids and not a.get('iscrowd',0)
                and not a['focus_ignore'] and lower<=a['area']<=upper for a in annotations))}
    output['per_class_AP']={c['name']:mean_valid(precision[:,:,ki,0,-1])
        for ki,c in enumerate(categories)}
    output['valid_gt']=sum(a['image_id'] in image_ids and not a.get('iscrowd',0)
        and not a['focus_ignore'] for a in annotations)
    output['images']=len(image_ids)
    output['max_detections_global_and_per_class']=cap
    output['metric_definition']='COCO 101-point AP, equal category mean, raw-image bbox area; not published VisDrone AP'
    output['pycocotools_version']=importlib.metadata.version('pycocotools')
    output['cocoeval_source_sha256']=sha256(Path(sys.modules['pycocotools.cocoeval'].__file__))
    return output


def load_visdrone(dataset,predictions):
    images,annotations,detections,prepared=[],[],[],[]
    names=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
    categories=[{'id':i+1,'name':name} for i,name in enumerate(names)]
    annotation_files=sorted((Path(dataset)/'annotations').glob('*.txt'))
    if not annotation_files:
        raise ValueError('No raw annotations found')
    annotation_hash=hashlib.sha256()
    detection_hash=hashlib.sha256()
    for image_id,ann in enumerate(annotation_files,1):
        prediction=Path(predictions)/ann.name
        if not prediction.exists():
            raise FileNotFoundError(prediction)
        for h,file in [(annotation_hash,ann),(detection_hash,prediction)]:
            h.update(ann.name.encode());h.update(b'\0');h.update(file.read_bytes());h.update(b'\0')
        gt,dt=read_rows(ann),read_rows(prediction)
        if len(dt):
            if not np.isfinite(dt).all() or np.any(dt[:-1,4]<dt[1:,4]):
                raise ValueError('Prediction rows must be finite and globally score sorted')
            if np.any(dt[:,2:4]<=0) or np.any((dt[:,4]<0)|(dt[:,4]>1)):
                raise ValueError('Invalid exported box or confidence')
            if np.any((dt[:,5]<1)|(dt[:,5]>10)|(dt[:,5]!=dt[:,5].astype(int))):
                raise ValueError('Invalid VisDrone class mapping')
        with Image.open(Path(dataset)/'images'/(ann.stem+'.jpg')) as image:
            width,height=image.size
        gt,dt=drop_ignored_regions(gt,dt,height,width)
        official_gt=gt.copy();official_gt[:,4]=1-official_gt[:,4]
        prepared.append((official_gt,dt))
        images.append({'id':image_id,'file_name':ann.stem+'.jpg','width':width,'height':height})
        for row in gt:
            if not 1<=row[5]<=10:
                continue
            if not np.isfinite(row).all() or np.any(row[2:4]<=0):
                raise ValueError('Invalid raw ground truth')
            annotations.append({'id':len(annotations)+1,'image_id':image_id,'category_id':int(row[5]),
                'bbox':row[:4].tolist(),'area':float(row[2]*row[3]),'iscrowd':int(row[4]==0),
                'occlusion':int(row[7]),'truncation':int(row[6])})
        for row in dt:
            detections.append({'image_id':image_id,'category_id':int(row[5]),
                'bbox':row[:4].tolist(),'score':float(row[4])})
    return images,annotations,categories,detections,prepared,{
        'annotations_sha256':annotation_hash.hexdigest(),'prediction_rows_sha256':detection_hash.hexdigest()}


def verified_author_metrics(prepared,parity_report):
    evidence=json.loads(Path(parity_report).read_text())
    if not evidence.get('passed') or evidence['python_evaluator_sha256']!=sha256(CORE/'visdrone_toolkit_port.py'):
        raise RuntimeError('Published-toolkit runtime parity missing or evaluator source changed')
    output=evaluate(prepared)
    output['runtime_parity_status']='Passed against original functions in GNU Octave'
    output['runtime_parity_evidence_sha256']=sha256(parity_report)
    return output
