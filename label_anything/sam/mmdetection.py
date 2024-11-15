# Copyright (c) OpenMMLab. All rights reserved.
import io
import json
import logging
import os
import random
import string
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
from botocore.exceptions import ClientError
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_single_tag_keys
from label_studio_tools.core.utils.io import get_data_dir

logger = logging.getLogger(__name__)


def load_sam_model(
        model_name="sam_hq",
        device="cuda:0",
        sam_config="vit_b",
        sam_checkpoint_file="sam_hq_vit_b.pth"
):
    """
    Loads the Segment Anything model on initializing Label studio, so if you call it outside MyModel it doesn't load every time you try to make a prediction
    Returns the predictor object. For more, look at Facebook's SAM docs
    """
    if model_name == "sam":
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except:
            raise ModuleNotFoundError(
                "segment_anything is not installed, run `pip install segment_anything` to install")
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    elif model_name == "sam_hq":
        from segment_anything_hq import sam_model_registry, SamPredictor
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    elif model_name == "mobile_sam":
        from models.mobile_sam import SamPredictor, sam_model_registry
        sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint_file)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    else:
        raise NotImplementedError


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection."""

    def __init__(self,
                 model_name="sam_hq",
                 sam_config='vit_b',
                 sam_checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 device='cpu',
                 **kwargs
        ):
        super(MMDetection, self).__init__(**kwargs)
        self.labels_file = labels_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
             self.parsed_label_config, 'KeyPointLabels', 'Image'
        )

        print('Load new model from: ', sam_config, sam_checkpoint_file)
        self.model = load_sam_model(model_name, device, sam_config, sam_checkpoint_file)

    def _get_image_url(self, task):
        image_url = task['data'].get(
            self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': key
                    })
            except ClientError as exc:
                logger.warning(
                    f'Can\'t generate presigned URL for {image_url}. Reason: {exc}'  # noqa E501
                )
        return image_url

    def predict(self, tasks, **kwargs):
        results = []
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)

        if kwargs.get('context') is None:
            return []

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.model.set_image(image)

        prompt_type = kwargs['context']['result'][0]['type']
        original_height = kwargs['context']['result'][0]['original_height']
        original_width = kwargs['context']['result'][0]['original_width']

        if prompt_type == 'keypointlabels':
            # getting x and y coordinates of the keypoint
            x = kwargs['context']['result'][0]['value']['x'] * original_width / 100
            y = kwargs['context']['result'][0]['value']['y'] * original_height / 100
            output_label = kwargs['context']['result'][0]['value']['labels'][0]

            masks, _, _ = self.model.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
        else:
            return []

        mask = masks[0].astype(np.uint8)  # each mask has shape [H, W]
        # converting the mask from the model to RLE format which is usable in Label Studio

        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 计算外接矩形
        new_contours = []
        for contour in contours:
            new_contours.extend(list(contour))
        new_contours = np.array(new_contours)
        x, y, w, h = cv2.boundingRect(new_contours)
        results.append({
            'from_name': self.from_name,
            'to_name': self.to_name,
            'type': 'keypointlabels',
            'value': {
                'keypointlabels': [output_label],
                'x': float(x) / original_width * 100,
                'y': float(y) / original_height * 100,
                'width': float(w) / original_width * 100,
                'height': float(h) / original_height * 100,
            },
            "id": ''.join(random.SystemRandom().choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6)), # creates a random ID for your label every time
        })

        return [{'result': results}]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
