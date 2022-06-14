import json
import os

import decord
import numpy as np

DATA_INFO_FILE = "data/annotations/data_info.json"
ORIG_FPS = 30
FPS = 4


def generate_dataset_info(dataset_name="THUMOS"):
    """TODO: Docstring for generate_dataset_info.

    Kwargs:
        dataset_name (string): TODO

    Returns: TODO

    """
    info = json.load(open(DATA_INFO_FILE, "r"))[dataset_name]
    class_names = info["class_names"]
    idx = list(range(len(class_names)))
    name_idx_dict = dict(zip(class_names, idx))
    return name_idx_dict


def parse_annotations(anno_file):
    """parse annotation file

    Args:
        anno_file (string): The annotation file.

    Returns: Dict.

    """
    annotations = {}
    for i, line in enumerate(open(anno_file, "r")):
        if i == 0:
            continue
        (
            video_name,
            class_name,
            idx,
            start,
            end,
            start_frame,
            end_frame,
        ) = line.strip().split(",")
        start_frame = int(float(start) * FPS)
        end_frame = int(float(end) * FPS)
        if video_name in annotations:
            annotations[video_name].append((start_frame, end_frame, class_name))
        else:
            annotations[video_name] = [(start_frame, end_frame, class_name)]
    return annotations


def generate_perframe_label(name_idx_dict, video_dir, annotations, dst_dir):
    num_classes = len(name_idx_dict)
    print(num_classes)

    for video_name, annos in annotations.items():
        video_path = os.path.join(video_dir, video_name + ".mp4")
        video = decord.VideoReader(video_path)
        num_frames = int(float(len(video)) / ORIG_FPS * FPS)
        perframe_gt = np.zeros([num_frames, num_classes], dtype=np.int64)
        for anno in annos:
            start, end, class_name = anno
            perframe_gt[start : end + 1, name_idx_dict[class_name]] = 1
        # save
        perframe_gt[np.where(np.sum(perframe_gt, axis=1) == 0)[0], 0] = 1
        dst_path = os.path.join(dst_dir, video_name + ".npy")
        np.save(dst_path, perframe_gt)


def main():
    name_idx_dict = generate_dataset_info()

    # validation
    annotation_file = "data/annotations/thumos/val_Annotation.csv"
    video_dir = "data/thumos/video/validation"
    dst_dir = f"data/thumos/target_perframe"
    os.makedirs(dst_dir, exist_ok=True)
    annotations = parse_annotations(annotation_file)
    print(len(annotations))
    generate_perframe_label(name_idx_dict, video_dir, annotations, dst_dir)

    # test
    annotation_file = "data/annotations/thumos/test_Annotation_fixed.csv"
    video_dir = "data/thumos/video/test"
    dst_dir = f"data/thumos/target_perframe"
    os.makedirs(dst_dir, exist_ok=True)
    annotations = parse_annotations(annotation_file)
    print(len(annotations))
    generate_perframe_label(name_idx_dict, video_dir, annotations, dst_dir)


if __name__ == "__main__":
    main()
