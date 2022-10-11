import math
import sys
from collections import namedtuple
import numpy as np
import tensorflow as tf

# For backwards compatibility with V1
tf.compat.v1.disable_eager_execution()

BoundingBox = namedtuple('BoundingBox', ['x1', 'x2', 'y1', 'y2'])


class MTCNN(object):

    def __init__(self, model_dir):
        self.graph = tf.Graph()
        self.graph.as_default()

        tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.session = tf.compat.v1.Session(config=tf_config)
        self.session.as_default()

        print('Loading face-detector model...')
        sys.path.append(model_dir)
        global detect_face
        import detect_face
        self.pnet, self.rnet, self.onet = \
            detect_face.create_mtcnn(self.session, model_dir)
        print('Loaded face-detector model')

    def face_detect(self, imgs):
        threshold = [0.45, 0.6, 0.7]
        factor = 0.709
        vmargin = 0.2582651235637604
        hmargin = 0.3449094129917718
        detection_window_size_ratio = .2

        detections = detect_face.bulk_detect_face(
            imgs, detection_window_size_ratio, self.pnet, self.rnet, self.onet,
            threshold, factor)

        batch_faces = []
        for img, bounding_boxes in zip(imgs, detections):
            if bounding_boxes is None:
                batch_faces.append([])
                continue
            frame_faces = []
            bounding_boxes = bounding_boxes[0]
            num_faces = bounding_boxes.shape[0]
            for i in range(num_faces):
                confidence = bounding_boxes[i][4]
                if confidence < .1:
                    continue

                img_size = np.asarray(img.shape)[0:2]
                det = np.squeeze(bounding_boxes[i][0:5])
                vmargin_pix = int((det[2] - det[0]) * vmargin)
                hmargin_pix = int((det[3] - det[1]) * hmargin)
                frame_faces.append({
                    'x1': np.maximum(det[0] - hmargin_pix / 2, 0) / img_size[1],
                    'y1': np.maximum(det[1] - vmargin_pix / 2, 0) / img_size[0],
                    'x2': np.minimum(det[2] + hmargin_pix / 2, img_size[1]) / img_size[1],
                    'y2': np.minimum(det[3] + vmargin_pix / 2, img_size[0]) / img_size[0],
                    'score': det[4]
                })

            batch_faces.append(frame_faces)
        return batch_faces

    def close(self):
        self.session.close()
