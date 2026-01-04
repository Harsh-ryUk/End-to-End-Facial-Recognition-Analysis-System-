import cv2
import numpy as np
import onnxruntime
from ..utils.helpers import distance2bbox, distance2kps

class SCRFD:
    """
    SCRFD: Efficient Face Detection (ONNX)
    """
    def __init__(self, model_path, input_size=(640, 640), conf_thres=0.5, iou_thres=0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.mean = 127.5
        self.std = 128.0
        self.center_cache = {}
        
        self.session = onnxruntime.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, image, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(image.shape[0:2][::-1])
        
        blob = cv2.dnn.blobFromImage(image, 1.0/self.std, input_size, (self.mean, self.mean, self.mean), swapRB=True)
        outputs = self.session.run(None, {self.input_name: blob})
        
        # Unpack outputs 
        # (Simplified logic assuming specific output order or iterating strides)
        # Note: ONNX output order can vary; standard SCRFD 2.5/10g usually follows stride order
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc] * stride
            kps_preds = outputs[idx + self.fmc * 2] * stride
            
            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
                self.center_cache[key] = anchor_centers
                
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
            
        return scores_list, bboxes_list, kpss_list

    def detect(self, image, max_num=0):
        # Resize logic
        width, height = self.input_size
        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = height / width
        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)
            
        det_scale = float(new_height) / image.shape[0]
        resized_image = cv2.resize(image, (new_width, new_height))
        det_image = np.zeros((height, width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image
        
        scores_list, bboxes_list, kpss_list = self.forward(det_image, self.conf_thres)
        
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, self.iou_thres)
        
        det = pre_det[keep, :]
        kpss = kpss[order, :, :][keep, :, :]
        
        if 0 < max_num < det.shape[0]:
            det = det[:max_num]
            kpss = kpss[:max_num]
            
        return det, kpss

    def nms(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
