from scannerpy import protobufs
from scannerpy.storage import NullElement
from scannerpy.types import BboxList


DILATE_AMOUNT = 1.05
def dilate_bboxes(config, bboxes: BboxList) -> BboxList:
    return [
        protobufs.BoundingBox(
            x1=bb.x1 * (2. - DILATE_AMOUNT),
            x2=bb.x2 * DILATE_AMOUNT,
            y1=bb.y1 * (2. - DILATE_AMOUNT),
            y2=bb.y2 * DILATE_AMOUNT,
            score=bb.score
        ) for bb in bboxes
    ]


def get_face_bboxes_results(detected_faces, stride):
    assert isinstance(stride, int)

    result = []  # [(<face_id>, {'frame_num': <n>, 'bbox': <bbox_dict>}), ...]
    frame_num = 0
    for faces in detected_faces:
        faces_in_frame = [
            (face_id, {'frame_num': frame_num, 'bbox': bbox_to_dict(face)})
            for face_id, face in enumerate(faces, len(result))
        ]

        result += faces_in_frame
        frame_num += stride

    return result


def get_face_embeddings_results(face_embeddings):
    result = []  # [(<face_id>, <embedding>), ...]
    for embeddings in face_embeddings:
        if isinstance(embeddings, NullElement):
            continue

        faces_in_frame = [
            (face_id, embed.tolist())
            for face_id, embed in enumerate(embeddings, len(result))
        ]

        result += faces_in_frame

    return result


def bbox_to_dict(b):
    return {'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2, 'score': b.score}
