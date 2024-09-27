import traceback
from typing import Optional, List, Tuple
import numpy as np
import supervisely as sly
from supervisely.app.widgets.sly_tqdm.sly_tqdm import CustomTqdm
from supervisely import ProjectMeta, ImageInfo, TagCollection
from supervisely import Bitmap, Rectangle, Polygon, Label
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.geometry.helpers import get_effective_nonoverlapping_masks
from supervisely.metric.matching import get_geometries_iou, match_indices_by_score
from supervisely.metric.common import (
    safe_ratio,
    TRUE_POSITIVE,
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    PRECISION,
    RECALL,
    F1_MEASURE,
    TOTAL_GROUND_TRUTH,
    TOTAL_PREDICTIONS,
)
from supervisely.metric.iou_metric import IOU, INTERSECTION, UNION


ERROR_PIXELS = "error-pixels"
TOTAL_PIXELS = "total-pixels"
PIXEL_ACCURACY = "pixel-accuracy"

NUM_OBJECTS_GT = "num-objects-gt"
NUM_OBJECTS_PRED = "num-objects-pred"
MATCHES_TRUE_POSITIVE = "matches-true-positive"
MATCHES_FALSE_POSITIVE = "matches-false-positive"
MATCHES_FALSE_NEGATIVE = "matches-false-negative"
MATCHES_PRECISION = "matches-precision"
MATCHES_RECALL = "matches-recall"
MATCHES_F1 = "matches-f1"

TAGS_TRUE_POSITIVE = "tags-true-positive"
TAGS_FALSE_POSITIVE = "tags-false-positive"
TAGS_FALSE_NEGATIVE = "tags-false-negative"
TAGS_PRECISION = "tags-precision"
TAGS_RECALL = "tags-recall"
TAGS_F1 = "tags-f1"
TAGS_TOTAL_GT = "tags-total-gt"
TAGS_TOTAL_PRED = "tags-total-pred"

_OBJ_MATCHES_METRIC_NAMES = {
    TRUE_POSITIVE: MATCHES_TRUE_POSITIVE,
    FALSE_POSITIVE: MATCHES_FALSE_POSITIVE,
    FALSE_NEGATIVE: MATCHES_FALSE_NEGATIVE,
    PRECISION: MATCHES_PRECISION,
    RECALL: MATCHES_RECALL,
    F1_MEASURE: MATCHES_F1,
    TOTAL_GROUND_TRUTH: NUM_OBJECTS_GT,
    TOTAL_PREDICTIONS: NUM_OBJECTS_PRED,
}

_TAG_METRIC_NAMES = {
    TRUE_POSITIVE: TAGS_TRUE_POSITIVE,
    FALSE_POSITIVE: TAGS_FALSE_POSITIVE,
    FALSE_NEGATIVE: TAGS_FALSE_NEGATIVE,
    PRECISION: TAGS_PRECISION,
    RECALL: TAGS_RECALL,
    F1_MEASURE: TAGS_F1,
    TOTAL_GROUND_TRUTH: TAGS_TOTAL_GT,
    TOTAL_PREDICTIONS: TAGS_TOTAL_PRED,
}

OVERALL_SCORE = "overall-score"


SEGMENTATION_GEOMETRIES = (Bitmap, Polygon)


class MetricsException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


class MetricValue:
    def __init__(self):
        self.value = 0
        self.metric_name = ""
        self.class_gt = ""
        self.image_gt_id = 0
        self.image_pred_id = 0
        self.tag_name = ""

    def to_json(self):
        return {
            "value": self.value,
            "metric_name": self.metric_name,
            "class_gt": self.class_gt,
            "image_gt_id": self.image_gt_id,
            "image_pred_id": self.image_pred_id,
            "tag_name": self.tag_name,
        }


class ComputeMetricsReq:
    def __init__(
        self,
        united_meta: ProjectMeta,
        img_infos_gt: List[ImageInfo],
        img_infos_pred: List[ImageInfo],
        ann_infos_gt: List[AnnotationInfo],
        ann_infos_pred: List[AnnotationInfo],
        class_mapping: dict,
        tags_whitelist: List[str],
        obj_tags_whitelist: List[str],
        iou_threshold: float,
        segmentation_mode: bool,
    ) -> None:
        self.united_meta = united_meta
        self.img_infos_gt = img_infos_gt
        self.img_infos_pred = img_infos_pred
        self.ann_infos_gt = ann_infos_gt
        self.ann_infos_pred = ann_infos_pred
        self.class_mapping = class_mapping
        self.tags_whitelist = tags_whitelist
        self.obj_tags_whitelist = obj_tags_whitelist
        self.iou_threshold = iou_threshold
        self.segmentation_mode = segmentation_mode


class ComputeMetricsResult:
    def __init__(self):
        self.image_metrics = []
        self.error_message = None

    def add(self, metric_value):
        self.image_metrics.append(metric_value)
        return self.image_metrics[-1]

    def to_json(self):
        if self.error_message is not None:
            return {"error": self.error_message}
        return [image_metric.to_json() for image_metric in self.image_metrics]


def _make_counters():
    return {TRUE_POSITIVE: 0, FALSE_POSITIVE: 0, FALSE_NEGATIVE: 0}


def _make_pixel_counters():
    return {INTERSECTION: 0, UNION: 0, ERROR_PIXELS: 0, TOTAL_PIXELS: 0}


def _fill_metric_value(
    metric_value,
    name,
    value,
    class_gt=None,
    image_gt_id=None,
    image_pred_id=None,
    tag_name=None,
):
    metric_value.value = value
    metric_value.metric_name = name
    if class_gt is not None:
        metric_value.class_gt = class_gt
    if image_gt_id is not None:
        metric_value.image_gt_id = image_gt_id
    if image_pred_id is not None:
        metric_value.image_pred_id = image_pred_id
    if tag_name is not None:
        metric_value.tag_name = tag_name


def _add_matching_metrics(
    dest,
    counters,
    metric_name_config,
    class_gt=None,
    image_gt_id=None,
    image_pred_id=None,
    tag_name=None,
):
    gt_total_key = metric_name_config[TOTAL_GROUND_TRUTH]
    pred_total_key = metric_name_config[TOTAL_PREDICTIONS]
    result_values = {
        metric_name_config[TRUE_POSITIVE]: counters[TRUE_POSITIVE],
        metric_name_config[FALSE_POSITIVE]: counters[FALSE_POSITIVE],
        metric_name_config[FALSE_NEGATIVE]: counters[FALSE_NEGATIVE],
        gt_total_key: counters[TRUE_POSITIVE] + counters[FALSE_NEGATIVE],
        pred_total_key: counters[TRUE_POSITIVE] + counters[FALSE_POSITIVE],
    }

    if result_values[gt_total_key] > 0:
        result_values[metric_name_config[RECALL]] = safe_ratio(
            counters[TRUE_POSITIVE], result_values[gt_total_key]
        )

    if result_values[pred_total_key] > 0:
        result_values[metric_name_config[PRECISION]] = safe_ratio(
            counters[TRUE_POSITIVE], result_values[pred_total_key]
        )

    if result_values[gt_total_key] > 0 or result_values[pred_total_key] > 0:
        result_values[metric_name_config[F1_MEASURE]] = (
            2
            * counters[TRUE_POSITIVE]
            / (2 * counters[TRUE_POSITIVE] + counters[FALSE_NEGATIVE] + counters[FALSE_POSITIVE])
        )

    for out_name, val in result_values.items():
        _fill_metric_value(
            dest.add(MetricValue()),
            out_name,
            val,
            class_gt=class_gt,
            image_gt_id=image_gt_id,
            image_pred_id=image_pred_id,
            tag_name=tag_name,
        )

    return result_values


def _add_pixel_metrics(dest, counters, class_gt, image_gt_id=None, image_pred_id=None):
    result_values = dict()
    if counters[TOTAL_PIXELS] > 0:
        result_values[PIXEL_ACCURACY] = 1.0 - counters[ERROR_PIXELS] / counters[TOTAL_PIXELS]
    if counters.get(UNION, 0) > 0:
        result_values[IOU] = counters[INTERSECTION] / counters[UNION]

    for out_name, val in result_values.items():
        _fill_metric_value(
            dest.add(MetricValue()),
            out_name,
            val,
            class_gt=class_gt,
            image_gt_id=image_gt_id,
            image_pred_id=image_pred_id,
        )
    return result_values


def _maybe_add_average_metric(dest, metrics, metric_name, image_gt_id=None, image_pred_id=None):
    values = [m[metric_name] for m in metrics if metric_name in m]
    if len(values) > 0:
        avg_metric = np.mean(values).item()
        _fill_metric_value(
            dest.add(MetricValue()),
            metric_name,
            avg_metric,
            image_gt_id=image_gt_id,
            image_pred_id=image_pred_id,
        )
        return {metric_name: avg_metric}
    else:
        return {}


def add_tag_counts(dest_counters, tags_gt, tags_pred, tags_whitelist, tp_key, fp_key, fn_key):
    effective_tags_gt = set((tag.name, tag.value) for tag in tags_gt if tag.name in tags_whitelist)
    effective_tags_pred = set(
        (tag.name, tag.value) for tag in tags_pred if tag.name in tags_whitelist
    )
    for name, _ in effective_tags_pred - effective_tags_gt:
        dest_counters[name][fp_key] += 1
    for name, _ in effective_tags_gt - effective_tags_pred:
        dest_counters[name][fn_key] += 1
    for name, _ in effective_tags_gt & effective_tags_pred:
        dest_counters[name][tp_key] += 1


def _sum_update_counters(dest, update, ignore_keys=None):
    for k, v in update.items():
        if ignore_keys is None or k in ignore_keys:
            dest[k] += v


def safe_get_geometries_iou(g1, g2):
    if g1 is None or g2 is None:
        return -1
    else:
        return get_geometries_iou(g1, g2)


def is_segmentation(label: Label, segmentation_mode: bool):
    if not segmentation_mode:
        return False
    if type(label.geometry) in SEGMENTATION_GEOMETRIES:
        return True


def compute_metrics(
    request: ComputeMetricsReq, progress: Optional[CustomTqdm] = None
) -> Tuple[ComputeMetricsResult, List[Bitmap]]:
    result = ComputeMetricsResult()
    iou_threshold = request.iou_threshold
    tags_whitelist = set(request.tags_whitelist)
    obj_tags_whitelist = set(request.obj_tags_whitelist)
    meta = request.united_meta
    img_infos_gt = request.img_infos_gt
    img_infos_pred = request.img_infos_pred
    ann_infos_gt = request.ann_infos_gt
    ann_infos_pred = request.ann_infos_pred
    class_mapping = request.class_mapping
    segmentation_mode = request.segmentation_mode
    difference_geometries = []
    try:
        if len(img_infos_gt) != len(img_infos_pred):
            raise MetricsException(
                message="Ground truth images infos and Prediction images infos have different lengths."
            )

        if len(ann_infos_gt) != len(ann_infos_pred):
            raise MetricsException(
                message="Ground truth annotations and Prediction annotations have different lengths."
            )

        class_matching_counters = {class_gt: _make_counters() for class_gt in class_mapping}
        class_pixel_counters = {
            class_mapping[class_gt]: _make_pixel_counters() for class_gt in class_mapping
        }
        tag_counters = {
            tag_name: _make_counters() for tag_name in set(tags_whitelist) | set(obj_tags_whitelist)
        }
        total_pixel_error = 0
        total_pixels = 0

        for (
            gt_image_info,
            pred_image_info,
            gt_ann_info,
            pred_ann_info,
        ) in zip(
            img_infos_gt,
            img_infos_pred,
            ann_infos_gt,
            ann_infos_pred,
        ):
            gt_ann = sly.Annotation.from_json(gt_ann_info.annotation, meta)
            pred_ann = sly.Annotation.from_json(pred_ann_info.annotation, meta)

            image_class_counters = {class_gt: _make_counters() for class_gt in class_mapping}
            image_pixel_counters = {class_gt: _make_pixel_counters() for class_gt in class_mapping}
            image_tag_counters = {tag_name: _make_counters() for tag_name in tag_counters}

            image_class_overall_counters = _make_counters()
            image_tag_overall_counters = _make_counters()

            add_tag_counts(
                image_tag_counters,
                gt_ann.img_tags,
                pred_ann.img_tags,
                tags_whitelist,
                tp_key=TRUE_POSITIVE,
                fp_key=FALSE_POSITIVE,
                fn_key=FALSE_NEGATIVE,
            )
            image_errors_canvas = np.zeros(gt_ann.img_size, dtype=np.bool)

            # non segmentation labels
            nonseg_gt_label_idxs = [
                idx
                for idx, label in enumerate(gt_ann.labels)
                if not is_segmentation(label, segmentation_mode)
            ]
            nonseg_pred_labels_idxs = [
                idx
                for idx, label in enumerate(pred_ann.labels)
                if not is_segmentation(label, segmentation_mode)
            ]

            nonseg_gt_class_to_indices = {}
            for idx in nonseg_gt_label_idxs:
                label = gt_ann.labels[idx]
                nonseg_gt_class_to_indices.setdefault(label.obj_class.name, []).append(idx)
            nonseg_pred_class_to_indices = {}
            for idx in nonseg_pred_labels_idxs:
                label = pred_ann.labels[idx]
                nonseg_pred_class_to_indices.setdefault(label.obj_class.name, []).append(idx)

            # segmentation labels
            seg_gt_label_idxs = [
                idx
                for idx, label in enumerate(gt_ann.labels)
                if is_segmentation(label, segmentation_mode)
            ]
            seg_pred_label_idxs = [
                idx
                for idx, label in enumerate(pred_ann.labels)
                if is_segmentation(label, segmentation_mode)
            ]

            seg_gt_class_to_indices = {}
            for idx in seg_gt_label_idxs:
                label = gt_ann.labels[idx]
                seg_gt_class_to_indices.setdefault(label.obj_class.name, []).append(idx)
            seg_pred_class_to_indices = {}
            for idx in seg_pred_label_idxs:
                label = pred_ann.labels[idx]
                seg_pred_class_to_indices.setdefault(label.obj_class.name, []).append(idx)

            # segementation masks and canvas for segmentation labels
            (
                seg_gt_effective_masks,
                seg_gt_effective_canvas,
            ) = get_effective_nonoverlapping_masks(
                [gt_ann.labels[idx].geometry for idx in seg_gt_label_idxs],
                img_size=gt_ann.img_size,
            )
            gt_label_idx_to_effective_mask_idx = {
                idx: effective_idx for effective_idx, idx in enumerate(seg_gt_label_idxs)
            }
            (
                seg_pred_effective_masks,
                seg_pred_effective_canvas,
            ) = get_effective_nonoverlapping_masks(
                [pred_ann.labels[idx].geometry for idx in seg_pred_label_idxs],
                img_size=pred_ann.img_size,
            )
            pred_label_idx_to_effective_mask_idx = {
                idx: effective_idx for effective_idx, idx in enumerate(seg_pred_label_idxs)
            }

            # iterating over classes
            for gt_class, pred_class in class_mapping.items():
                image_class_pixel_counters = image_pixel_counters[gt_class]
                this_image_class_counters = image_class_counters[gt_class]

                # Non segmentation labels

                # Get indices of labels of matching classes
                gt_class_indices = nonseg_gt_class_to_indices.get(gt_class, [])
                pred_class_indices = nonseg_pred_class_to_indices.get(pred_class, [])
                matching_results = match_indices_by_score(
                    [gt_ann.labels[idx].geometry for idx in gt_class_indices],
                    [pred_ann.labels[idx].geometry for idx in pred_class_indices],
                    iou_threshold,
                    safe_get_geometries_iou,
                )

                # Object matching counters
                this_image_class_counters[TRUE_POSITIVE] = len(matching_results.matches)
                this_image_class_counters[FALSE_NEGATIVE] = len(
                    matching_results.unmatched_indices_1
                )
                this_image_class_counters[FALSE_POSITIVE] = len(
                    matching_results.unmatched_indices_2
                )

                # Tags counters
                for match in matching_results.matches:
                    add_tag_counts(
                        image_tag_counters,
                        gt_ann.labels[gt_class_indices[match.idx_1]].tags,
                        pred_ann.labels[pred_class_indices[match.idx_2]].tags,
                        obj_tags_whitelist,
                        tp_key=TRUE_POSITIVE,
                        fp_key=FALSE_POSITIVE,
                        fn_key=FALSE_NEGATIVE,
                    )
                for fn_label_idx in matching_results.unmatched_indices_1:
                    add_tag_counts(
                        image_tag_counters,
                        gt_ann.labels[gt_class_indices[fn_label_idx]].tags,
                        TagCollection(),
                        obj_tags_whitelist,
                        tp_key=TRUE_POSITIVE,
                        fp_key=FALSE_POSITIVE,
                        fn_key=FALSE_NEGATIVE,
                    )
                for fp_label_idx in matching_results.unmatched_indices_2:
                    add_tag_counts(
                        image_tag_counters,
                        TagCollection(),
                        pred_ann.labels[pred_class_indices[fp_label_idx]].tags,
                        obj_tags_whitelist,
                        tp_key=TRUE_POSITIVE,
                        fp_key=FALSE_POSITIVE,
                        fn_key=FALSE_NEGATIVE,
                    )

                # Iterating over matched labels
                for gt_match, pred_match, _ in matching_results.matches:
                    gt_match_idx = gt_class_indices[gt_match]
                    pred_match_idx = pred_class_indices[pred_match]
                    gt_geometry = gt_ann.labels[gt_match_idx].geometry
                    pred_geometry = pred_ann.labels[pred_match_idx].geometry
                    gt_canvas = np.zeros(gt_ann.img_size, dtype=np.bool)
                    gt_geometry.draw(gt_canvas, color=True)
                    pred_canvas = np.zeros(gt_ann.img_size, dtype=np.bool)
                    pred_geometry.draw(pred_canvas, color=True)
                    error_canvas = gt_canvas != pred_canvas

                    # Pixel counters
                    image_class_pixel_counters[INTERSECTION] += np.sum(
                        gt_canvas & pred_canvas
                    ).item()
                    image_class_pixel_counters[UNION] += np.sum(gt_canvas | pred_canvas).item()
                    image_class_pixel_counters[ERROR_PIXELS] += np.sum(error_canvas).item()
                    common_bbox_area = int(
                        Rectangle.from_geometries_list([gt_geometry, pred_geometry]).area
                    )
                    image_class_pixel_counters[TOTAL_PIXELS] += common_bbox_area

                    # Add errors to image errors canvas
                    image_errors_canvas |= error_canvas

                # Add errors to image errors canvas for unmatched labels
                for gt_idx in matching_results.unmatched_indices_1:
                    gt_geometry = gt_ann.labels[gt_class_indices[gt_idx]].geometry
                    gt_canvas = np.zeros(gt_ann.img_size, dtype=np.bool)
                    gt_geometry.draw(gt_canvas, color=True)
                    image_errors_canvas |= gt_canvas
                for pred_idx in matching_results.unmatched_indices_2:
                    pred_geometry = pred_ann.labels[pred_class_indices[pred_idx]].geometry
                    pred_canvas = np.zeros(gt_ann.img_size, dtype=np.bool)
                    pred_geometry.draw(pred_canvas, color=True)
                    image_errors_canvas |= pred_canvas

                # Segmentation labels
                gt_class_indices = seg_gt_class_to_indices.get(gt_class, [])
                pred_class_indices = seg_pred_class_to_indices.get(pred_class, [])
                seg_class_masks_gt = [
                    seg_gt_effective_masks[gt_label_idx_to_effective_mask_idx[idx]]
                    for idx in gt_class_indices
                ]
                seg_class_masks_pred = [
                    seg_pred_effective_masks[pred_label_idx_to_effective_mask_idx[idx]]
                    for idx in pred_class_indices
                ]
                matching_results = match_indices_by_score(
                    seg_class_masks_gt,
                    seg_class_masks_pred,
                    iou_threshold,
                    safe_get_geometries_iou,
                )

                # Object matching counters
                this_image_class_counters[TRUE_POSITIVE] += len(matching_results.matches)
                this_image_class_counters[FALSE_NEGATIVE] += len(
                    matching_results.unmatched_indices_1
                )
                this_image_class_counters[FALSE_POSITIVE] += len(
                    matching_results.unmatched_indices_2
                )

                # Tags counters
                for match in matching_results.matches:
                    add_tag_counts(
                        image_tag_counters,
                        gt_ann.labels[gt_class_indices[match.idx_1]].tags,
                        pred_ann.labels[pred_class_indices[match.idx_2]].tags,
                        obj_tags_whitelist,
                        tp_key=TRUE_POSITIVE,
                        fp_key=FALSE_POSITIVE,
                        fn_key=FALSE_NEGATIVE,
                    )

                for fn_label_idx in matching_results.unmatched_indices_1:
                    add_tag_counts(
                        image_tag_counters,
                        gt_ann.labels[gt_class_indices[fn_label_idx]].tags,
                        TagCollection(),
                        obj_tags_whitelist,
                        tp_key=TRUE_POSITIVE,
                        fp_key=FALSE_POSITIVE,
                        fn_key=FALSE_NEGATIVE,
                    )

                for fp_label_idx in matching_results.unmatched_indices_2:
                    add_tag_counts(
                        image_tag_counters,
                        TagCollection(),
                        pred_ann.labels[pred_class_indices[fp_label_idx]].tags,
                        obj_tags_whitelist,
                        tp_key=TRUE_POSITIVE,
                        fp_key=FALSE_POSITIVE,
                        fn_key=FALSE_NEGATIVE,
                    )

                gt_canvas = np.isin(seg_gt_effective_canvas, gt_class_indices)
                pred_canvas = np.isin(seg_pred_effective_canvas, pred_class_indices)
                error_canvas = gt_canvas != pred_canvas

                # Pixel counters
                image_class_pixel_counters[INTERSECTION] += np.sum(gt_canvas & pred_canvas).item()
                image_class_pixel_counters[UNION] += np.sum(gt_canvas | pred_canvas).item()
                image_class_pixel_counters[ERROR_PIXELS] += np.sum(error_canvas).item()
                image_class_pixel_counters[TOTAL_PIXELS] += error_canvas.size

                # Add nonrectangles differences to image errors canvas
                image_errors_canvas |= error_canvas

                # Update image class counters
                _sum_update_counters(class_pixel_counters[gt_class], image_class_pixel_counters)
                _sum_update_counters(class_matching_counters[gt_class], this_image_class_counters)
                _sum_update_counters(image_class_overall_counters, this_image_class_counters)

            # Update tag counters
            for tag_name, this_tag_counters in image_tag_counters.items():
                _sum_update_counters(tag_counters[tag_name], this_tag_counters)
                _sum_update_counters(image_tag_overall_counters, this_tag_counters)

            image_overall_score_components = {}

            # Object matching stats per image and class.
            for class_gt, this_class_counters in image_class_counters.items():
                _add_matching_metrics(
                    result,
                    this_class_counters,
                    metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                    class_gt=class_gt,
                    image_gt_id=gt_image_info.id,
                    image_pred_id=pred_image_info.id,
                )
            image_class_overall_metrics = _add_matching_metrics(
                result,
                image_class_overall_counters,
                metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                image_gt_id=gt_image_info.id,
                image_pred_id=pred_image_info.id,
            )
            if MATCHES_F1 in image_class_overall_metrics:
                image_overall_score_components.update(
                    {MATCHES_F1: image_class_overall_metrics[MATCHES_F1]}
                )

            # Pixel accuracy metrics per image and class
            per_image_class_accuracy_metrics = {
                class_gt: _add_pixel_metrics(
                    result,
                    image_class_pixel_counters,
                    class_gt,
                    image_gt_id=gt_image_info.id,
                    image_pred_id=pred_image_info.id,
                )
                for class_gt, image_class_pixel_counters in image_pixel_counters.items()
            }
            image_overall_score_components.update(
                _maybe_add_average_metric(
                    result,
                    per_image_class_accuracy_metrics.values(),
                    IOU,
                    image_gt_id=gt_image_info.id,
                    image_pred_id=pred_image_info.id,
                )
            )

            image_pixel_error = np.sum(image_errors_canvas).item()
            num_image_pixels = image_errors_canvas.size
            _fill_metric_value(
                result.add(MetricValue()),
                PIXEL_ACCURACY,
                1.0 - image_pixel_error / num_image_pixels,
                image_gt_id=gt_image_info.id,
                image_pred_id=pred_image_info.id,
            )
            total_pixel_error += image_pixel_error
            total_pixels += num_image_pixels

            # Matching stats per image and tag.
            for tag_name, this_tag_counters in tag_counters.items():
                _add_matching_metrics(
                    result,
                    this_tag_counters,
                    metric_name_config=_TAG_METRIC_NAMES,
                    image_gt_id=gt_image_info.id,
                    image_pred_id=pred_image_info.id,
                    tag_name=tag_name,
                )
            image_tag_overall_metrics = _add_matching_metrics(
                result,
                image_tag_overall_counters,
                metric_name_config=_TAG_METRIC_NAMES,
                image_gt_id=gt_image_info.id,
                image_pred_id=pred_image_info.id,
            )
            if TAGS_F1 in image_tag_overall_metrics:
                image_overall_score_components.update({TAGS_F1: image_tag_overall_metrics[TAGS_F1]})

            if len(image_overall_score_components) > 0:
                overall_score = np.mean(list(image_overall_score_components.values())).item()
                _fill_metric_value(
                    result.add(MetricValue()),
                    OVERALL_SCORE,
                    overall_score,
                    image_gt_id=gt_image_info.id,
                    image_pred_id=pred_image_info.id,
                )

            if np.any(image_errors_canvas):
                difference_geometries.append(Bitmap(image_errors_canvas))
            else:
                difference_geometries.append(None)

            if progress is not None:
                progress.update(1)

        overall_score_components = {}

        # Overall metrics per class
        per_class_metrics = {
            class_gt: _add_matching_metrics(
                result,
                this_class_counters,
                metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                class_gt=class_gt,
            )
            for class_gt, this_class_counters in class_matching_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(result, per_class_metrics.values(), MATCHES_F1)
        )

        # Per class pixel accuracy metrics.
        per_class_accuracy_metrics = {
            class_gt: _add_pixel_metrics(result, image_class_pixel_counters, class_gt)
            for class_gt, image_class_pixel_counters in class_pixel_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(result, per_class_accuracy_metrics.values(), IOU)
        )
        if total_pixels > 0:
            _fill_metric_value(
                result.add(MetricValue()),
                PIXEL_ACCURACY,
                1.0 - total_pixel_error / total_pixels,
            )

        per_tag_metrics = {
            tag_name: _add_matching_metrics(
                result,
                this_tag_counters,
                metric_name_config=_TAG_METRIC_NAMES,
                tag_name=tag_name,
            )
            for tag_name, this_tag_counters in tag_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(result, per_tag_metrics.values(), TAGS_F1)
        )

        if len(overall_score_components) > 0:
            overall_score = np.mean(list(overall_score_components.values())).item()
            _fill_metric_value(result.add(MetricValue()), OVERALL_SCORE, overall_score)

    except MetricsException as exc:
        # Reset the result to make sure there is no incomplete data there
        result = ComputeMetricsResult()
        result.error_message = exc.message
        difference_geometries = []
        if progress is not None:
            progress.update(len(img_infos_gt))
    except Exception as exc:
        result = ComputeMetricsResult()
        result.error_message = "Unexpected exception: {}".format(traceback.format_exc())
        difference_geometries = []
        if progress is not None:
            progress.update(len(img_infos_gt))
    return result, difference_geometries


@sly.timeit
def calculate_exam_report(
    united_meta: ProjectMeta,
    img_infos_gt: List[ImageInfo],
    img_infos_pred: List[ImageInfo],
    ann_infos_gt: List[AnnotationInfo],
    ann_infos_pred: List[AnnotationInfo],
    class_mapping: dict,
    tags_whitelist: List[str],
    obj_tags_whitelist: List[str],
    iou_threshold: float,
    progress: Optional[CustomTqdm] = None,
    segmentation_mode: Optional[bool] = True,
) -> Tuple[List, List[Bitmap]]:
    request = ComputeMetricsReq(
        united_meta=united_meta,
        img_infos_gt=img_infos_gt,
        img_infos_pred=img_infos_pred,
        ann_infos_gt=ann_infos_gt,
        ann_infos_pred=ann_infos_pred,
        class_mapping=class_mapping,
        tags_whitelist=tags_whitelist,
        obj_tags_whitelist=obj_tags_whitelist,
        iou_threshold=iou_threshold,
        segmentation_mode=segmentation_mode,
    )
    result, diff_bitmaps = compute_metrics(request, progress)
    return result.to_json(), diff_bitmaps
