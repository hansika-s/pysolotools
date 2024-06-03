import argparse
import json
import logging
import multiprocessing
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import math
import random
from multiprocessing import Pool

import cv2
import numpy as np
from pycocotools.mask import encode as mask_to_rle

from pysolotools.constants import COCO_KEYPOINTS
from pysolotools.consumers import Solo
from pysolotools.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox2DAnnotationDefinition,
    Frame,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    KeypointAnnotationDefinition,
    RGBCameraCapture,
    SemanticSegmentationAnnotation,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SOLO2COCOConverter:
    """Convert SOLO to COCO format.
    This converter convert solo format to coco format.
    It supports 4 annotations 2d bbox,keypoints,instance
    and semantic segmentation. Based on annotation types
    of input solo data it does the conversion into coco.


        Examples:
        >>> solo=Solo("src_data_path")
        >>> converter = SOLO2COCOConverter(solo=solo)
        >>> converter.convert(output_path="output/data/path", dataset_name="coco")

    Expected output directory :
    coco:
        └── annotations
            ├── file_name.json
        └── images
    """

    def __init__(self, solo: Solo):
        self._solo = solo
        self._pool = multiprocessing.Pool(
            processes=1)  # multiprocessing.cpu_count())
        self._bbox_annotations = []
        self._instance_annotations = []
        self._semantic_annotations = []
        self._images = []

    def get_id(self) -> str:
        return "pysolo_extensions.converters.solo2coco"

    @staticmethod
    def _process_rgb_image(image_id, rgb_capture, output, data_root, sequence_num):
        """
        Args:
            image_id (int): Image ID
            rgb_capture (RGBCameraCapture): RGBCameraCapture object
            output (str): Output directory path
            data_root (str): Data root path
            sequence_num (int): Sequence number

        Returns:
            record: COCO format image record
        """
        width, height = rgb_capture.dimension

        image_to_folder = Path(output) / "images"
        image_to_folder.mkdir(parents=True, exist_ok=True)
        image_file = os.path.join(
            data_root, f"sequence.{sequence_num}/{rgb_capture.filename}"
        )

        image_to_file = f"camera_{image_id}.png"
        image_to = image_to_folder / image_to_file
        shutil.copy(str(image_file), str(image_to))

        record = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_to_file,
        }
        return record

    @staticmethod
    def _load_segmentation_image(filename, data_root, sequence_num):
        """
        Args:
            filename (str): Image file name
            data_root (str): Data root
            sequence_num (int): Sequence number

        Returns:
            img: Segmentation image

        """
        file_path = os.path.join(
            data_root, f"sequence.{sequence_num}", filename)
        img = cv2.imread(file_path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    @staticmethod
    def _compute_segmentation_map(ins_image=None, color=None):
        """
        Args:
            ins_image (): Segmentation image
            color (): RGBA value

        Returns:
            segmentation: Segmentation map
        """

        if ins_image.shape[-1] == 4:
            ins_color = (color[0], color[1], color[2], color[3])
        else:
            ins_color = (color[0], color[1], color[2])

        ins_mask = (ins_image == ins_color).prod(axis=-1).astype(np.uint8)
        segmentation = mask_to_rle(np.asfortranarray(ins_mask))
        segmentation["counts"] = segmentation["counts"].decode()

        return segmentation

    @staticmethod
    def _filter_annotation(ann_type, annotations):
        filtered_ann = list(
            filter(
                lambda k: isinstance(
                    k,
                    ann_type,
                ),
                annotations,
            )
        )
        if filtered_ann:
            return filtered_ann[0]
        return filtered_ann

    @staticmethod
    def _keypoints_map(solo_kp_map: Dict, kp_ann: KeypointAnnotation):
        """
        Args:
            solo_kp_map (Dict): SOLO keypoint dictionary that maps index to labels
            kp_ann (KeypointAnnotation): KeypointAnnotation object

        Returns:
            kp_map: Dict containing mapping of instance id to COCO format keypoints value and num keypoints tuple.
        """
        keypoint_map = {}
        for ann_kpt in kp_ann.values:
            keypoints_vals, num_keypoints = [], 0
            for k in COCO_KEYPOINTS:
                for kpt in ann_kpt.keypoints:
                    label = solo_kp_map[kpt.index]
                    if label == k:
                        keypoints_vals.extend(
                            [
                                int(np.floor(kpt.location[0])),
                                int(np.floor(kpt.location[1])),
                                kpt.state,
                            ]
                        )
                        if int(kpt.state) != 0:
                            num_keypoints += 1
                        break

            keypoint_map[ann_kpt.instanceId] = (
                num_keypoints,
                keypoints_vals,
            )
        return keypoint_map

    @staticmethod
    def _sem_class_color_map(sem_seg: SemanticSegmentationAnnotation):
        """
        Args:
            sem_seg (SemanticSegmentationAnnotation): SemanticSegmentationAnnotation object.

        Returns:
            class_color: Dict that maps class to its pixel value.
        """
        class_color = {}
        for ann_seg in sem_seg.instances:
            class_color[ann_seg.labelName] = ann_seg.pixelValue
        return class_color

    @staticmethod
    def _process_annotations(
        output, image_id, rgb_capture, sequence_num, data_root, solo_kp_map, interaction_metadata
    ):
        bbox_annotations = []
        instance_annotations = []
        semantic_annotations = []

        bbox_ann = SOLO2COCOConverter._filter_annotation(
            ann_type=BoundingBox2DAnnotation, annotations=rgb_capture.annotations
        ).values

        ins_seg = SOLO2COCOConverter._filter_annotation(
            ann_type=InstanceSegmentationAnnotation, annotations=rgb_capture.annotations
        )

        sem_seg = SOLO2COCOConverter._filter_annotation(
            ann_type=SemanticSegmentationAnnotation, annotations=rgb_capture.annotations
        )

        kp_ann = SOLO2COCOConverter._filter_annotation(
            ann_type=KeypointAnnotation, annotations=rgb_capture.annotations
        )
        if kp_ann:
            keypoint_map = SOLO2COCOConverter._keypoints_map(
                solo_kp_map, kp_ann)
        else:
            keypoint_map = {}

        if sem_seg:
            class_color_map = SOLO2COCOConverter._sem_class_color_map(sem_seg)
            sem_seg_img = SOLO2COCOConverter._load_segmentation_image(
                sem_seg.filename, data_root, sequence_num
            )

            for bbox in bbox_ann:
                x, y, w, h = [
                    int(bbox.origin[0]),
                    int(bbox.origin[1]),
                    int(bbox.dimension[0]),
                    int(bbox.dimension[1]),
                ]
                segmentation = []
                if bbox.labelName in class_color_map:
                    segmentation = SOLO2COCOConverter._compute_segmentation_map(
                        ins_image=sem_seg_img.copy(),
                        color=class_color_map[bbox.labelName],
                    )
                keypoints_vals, num_keypoints = [], 0
                if kp_ann and bbox.instanceId in keypoint_map:
                    num_keypoints, keypoints_vals = keypoint_map[bbox.instanceId]

                record = {
                    "segmentation": segmentation,
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "keypoints": keypoints_vals,
                    "num_keypoints": num_keypoints,
                    "category_id": bbox.labelId,
                    "id": bbox.instanceId,
                }
                semantic_annotations.append(record)

        if ins_seg:

            ins_seg_img = SOLO2COCOConverter._load_segmentation_image(
                ins_seg.filename, data_root, sequence_num
            )
            # Saving instance masks and depth maps
            save_masks_and_depth(image_id, ins_seg.filename, output,
                                 data_root, sequence_num)
            ins_seg_instances = ins_seg.instances

            if len(bbox_ann) != len(ins_seg_instances):
                raise ValueError(
                    "All bounding boxes does not have segmentation map.")

            # Added to map label_id and the bbox to fetch bbox of the active object
            label_to_bboxmap = dict(
                [(element.labelId, [element.origin[0], element.origin[1], element.dimension[0], element.dimension[1]]) for element in bbox_ann])
            label_to_bboxmap[-1] = [-1, -1, -1, -1]

            for bbox, ins in zip(bbox_ann, ins_seg_instances):

                x, y, w, h = [
                    int(bbox.origin[0]),
                    int(bbox.origin[1]),
                    int(bbox.dimension[0]),
                    int(bbox.dimension[1]),
                ]
                segmentation = SOLO2COCOConverter._compute_segmentation_map(
                    ins_image=ins_seg_img.copy(), color=ins.color
                )
                keypoints_vals, num_keypoints = [], 0
                if kp_ann and bbox.instanceId in keypoint_map:
                    num_keypoints, keypoints_vals = keypoint_map[bbox.instanceId]

                record = {
                    "segmentation": segmentation,
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "keypoints": keypoints_vals,
                    "num_keypoints": num_keypoints,
                    "category_id": bbox.labelId,
                    "id": bbox.instanceId,
                    "instance_id_color": {'r': ins.color[0], 'g': ins.color[1], 'b': ins.color[2], 'a': ins.color[3]},
                }
                record.update(interaction_metadata[str(bbox.instanceId)])
                try:
                    record["bbox_obj"] = label_to_bboxmap[record['id_obj']]
                except KeyError:  # In cases where the object is occluded, the bbox of the active object is not fetched
                    record["bbox_obj"] = [-1, -1, -1, -1]

                instance_annotations.append(record)

        else:

            for bbox in bbox_ann:
                x, y, w, h = [
                    int(bbox.origin[0]),
                    int(bbox.origin[1]),
                    int(bbox.dimension[0]),
                    int(bbox.dimension[1]),
                ]
                keypoints_vals, num_keypoints = [], 0
                if kp_ann and bbox.instanceId in keypoint_map:
                    num_keypoints, keypoints_vals = keypoint_map[bbox.instanceId]

                record = {
                    "segmentation": [],
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "keypoints": keypoints_vals,
                    "num_keypoints": num_keypoints,
                    "category_id": bbox.labelId,
                    "id": bbox.instanceId,
                }
                bbox_annotations.append(record)

        return bbox_annotations, instance_annotations, semantic_annotations

    def _categories(self):
        categories = []
        ann_defs = self._solo.annotation_definitions.annotationDefinitions
        bbox_ann_def = list(
            filter(
                lambda ann_def: isinstance(
                    ann_def, BoundingBox2DAnnotationDefinition),
                ann_defs,
            )
        )[0]
        for label_spec in bbox_ann_def.spec:
            record = {
                "id": label_spec.label_id,
                "name": label_spec.label_name,
                "supercategory": "default",
                "keypoints": [],
                "skeleton": [],
            }
            categories.append(record)

        return categories

    @staticmethod
    def _create_ann_file(data, output_dir, file_name):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = f"{output_dir}/{file_name}"
        with open(output_file, "w") as out:
            json.dump(data, out, indent=3)

    def _generate_hands_data(self, instances):
        hands_data = {
            "info": instances["info"],
            "licences": instances["licences"],
            "images": instances["images"],
            "annotations": [],
            "categories": []
        }

        ctr = 0
        for annotation in instances["annotations"]:
            if annotation["hand_side"] in [0, 1]:
                record = {
                    "id": ctr,
                    "image_id": annotation["image_id"],
                    "category_id": 0,
                    "bbox": annotation["bbox"],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"],
                    "hand_side": annotation["hand_side"],
                    "contact_state": annotation["contact_state"],
                    "bbox_obj": [] if annotation["bbox_obj"] == [-1, -1, -1, -1] else annotation["bbox_obj"],
                    "category_id_obj": annotation["id_obj"],
                }
                if annotation["contact_state"] == 1:
                    bbox = annotation['bbox_obj']
                    area = bbox[2] * bbox[3]
                    record["area_obj"] = area

                hands_data["annotations"].append(record)
                ctr += 1

        for category in instances["categories"]:
            if category['name'] == 'hand':
                hands_data["categories"].append({
                    "id": 0,
                    "name": category["name"],
                    "supercategory": category["supercategory"]
                })

        return hands_data

    def _write_out_annotations(self, output_dir, data_type):
        date_time = datetime.now()
        date_created = date_time.strftime("%A, %d %B, %Y")
        instances = {
            "info": {
                "year": datetime.now().year,
                "version": "0.0.1",
                "description": "COCO compatible Synthetic Dataset",
                "contributor": "Anonymous",
                "url": "Not Set",
                "date_created": date_created,
            },
            "licences": [{"id": 0, "name": "No License", "url": "Not Set"}],
            "images": self._images,
            "annotations": [],
            "categories": self._categories(),
        }
        if self._semantic_annotations:
            for idx, ann in enumerate(self._semantic_annotations):
                ann["id"] = idx
            instances["annotations"] = self._semantic_annotations
            self._create_ann_file(instances, output_dir, "semantic.json")

        if self._instance_annotations:
            for idx, ann in enumerate(self._instance_annotations):
                ann["id"] = idx
            instances["annotations"] = self._instance_annotations
            
            filename = f"{data_type}.json"

            self._create_ann_file(instances, output_dir, filename)

            if data_type in ['val', 'test']:
                hands_data = self._generate_hands_data(instances)
                hands_filename = f"{data_type}_hands.json"
                self._create_ann_file(hands_data, output_dir, hands_filename)

        else:
            for idx, ann in enumerate(self._bbox_annotations):
                ann["id"] = idx
            instances["annotations"] = self._bbox_annotations
            self._create_ann_file(instances, output_dir, "bbox.json")

    @staticmethod
    def _process_instances(
        frame: Frame, idx, output, data_root, solo_kp_map
    ) -> Tuple[Dict, List, List, List]:
        logger.info(f"Processing Frame number: {idx}")
        image_id = idx
        sequence_num = frame.sequence
        rgb_capture = list(
            filter(lambda cap: isinstance(
                cap, RGBCameraCapture), frame.captures)
        )[0]
        interaction_metadata = [
            metric for metric in frame.metrics if metric['id'] == 'metadata'][0]
        interaction_metadata = interaction_metadata['values'][0]['instances']
        interaction_metadata = dict(
            [(element['instanceId'], element['hand_object_interaction']) for element in interaction_metadata])

        img_record = SOLO2COCOConverter._process_rgb_image(
            image_id, rgb_capture, output, data_root, sequence_num
        )
        (
            ann_record,
            ins_ann_record,
            sem_ann_record,
        ) = SOLO2COCOConverter._process_annotations(
            output, image_id, rgb_capture, sequence_num, data_root, solo_kp_map, interaction_metadata
        )

        return img_record, ann_record, ins_ann_record, sem_ann_record

    def _get_solo_kp_map(self):
        solo_kp_map = {}
        kp_ann_def = list(
            filter(
                lambda k: isinstance(
                    k,
                    KeypointAnnotationDefinition,
                ),
                self._solo.annotation_definitions.annotationDefinitions,
            )
        )
        if kp_ann_def:
            for kp in kp_ann_def[0].template.keypoints:
                solo_kp_map[kp.index] = kp.label
        return solo_kp_map

    def callback(self, result):
        self._images.append(result[0])
        self._bbox_annotations += result[1]
        self._instance_annotations += result[2]
        self._semantic_annotations += result[3]

    def _process_split(self, split, output, split_type):
        solo_kp_map = self._get_solo_kp_map()

        with Pool() as pool:
            for idx, frame in split:
                pool.apply_async(
                self._process_instances,
                args=(frame, idx, output, self._solo.data_path, solo_kp_map),
                callback=self.callback,
            )
            pool.close()
            pool.join()

        self._write_out_annotations(output, data_type=split_type)
        self._clear_annotations()
    
    def _clear_annotations(self):
        self._images = []
        self._bbox_annotations = []
        self._instance_annotations = []
        self._semantic_annotations = []

    def convert(self, output_path: str, dataset_name: str = "coco"):
        output = os.path.join(output_path, dataset_name)

        frames_list = list(enumerate(self._solo.frames()))
        random.shuffle(frames_list)

        # train, val and test split
        train_ratio = 0.8
        val_ratio = 0.05

        train_split = frames_list[: int(len(frames_list) * train_ratio)]
        val_split = frames_list[int(len(
            frames_list) * train_ratio): int(len(frames_list) * (train_ratio + val_ratio))]
        test_split = frames_list[int(
            len(frames_list) * (train_ratio + val_ratio)):]
        
        self._process_split(train_split, output, 'train')
        self._process_split(val_split, output, 'val')
        self._process_split(test_split, output, 'test')

def save_masks_and_depth(image_id, filename, output, data_root, sequence_num):
    """Saves instance masks and depth maps from a sequence.

    Args:
        image_id (int): Image ID
        filename: Name of the instance segmentation image file.
        output (str): Output directory path
        data_root (str): Data root path
        sequence_num (int): Sequence number
    """

    os.makedirs(os.path.join(output, "instance_masks"), exist_ok=True)
    os.makedirs(os.path.join(output, "depth_maps"), exist_ok=True)

    sequence_path = f"sequence.{sequence_num}"

    mask_file_path = os.path.join(data_root, sequence_path, filename)
    depth_file_path = os.path.join(
        data_root, sequence_path, filename.replace("instance segmentation.png", "Depth.exr"))

    mask_save_path = os.path.join(
        output, "instance_masks", f"mask_{image_id}.png")
    depth_save_path = os.path.join(output, "depth_maps", f"map_{image_id}.png")

    shutil.copy(mask_file_path, mask_save_path)
    shutil.copy(depth_file_path, depth_save_path)


def cli():
    parser = argparse.ArgumentParser(
        prog="solo2coco",
        description=("Converts SOLO datasets into COCO datasets",),
        epilog="\n",
    )

    parser.add_argument("solo_path")
    parser.add_argument("coco_path")
    parser.add_argument(
        "-n", "--name", default="coco", help="The name of the coco dataset"
    )

    args = parser.parse_args(sys.argv[1:])

    solo = Solo(args.solo_path)

    converter = SOLO2COCOConverter(solo)

    converter.convert(output_path=args.coco_path, dataset_name=args.name)


if __name__ == "__main__":
    cli()
