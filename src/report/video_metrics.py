import traceback
from typing import Optional, List, Tuple
import numpy as np
import supervisely as sly
from supervisely.app.widgets.sly_tqdm.sly_tqdm import CustomTqdm
from supervisely import VideoAnnotation, VideoFigure, TagCollection
from supervisely.api.video.video_api import VideoInfo
from supervisely import Bitmap, Rectangle, Polygon
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
        self.gt_frame_n = 0
        self.pred_frame_n = 0
        self.tag = ""

    def to_json(self):
        return {
            "value": self.value,
            "metric_name": self.metric_name,
            "class_gt": self.class_gt,
            "gt_frame_n": self.gt_frame_n,
            "pred_frame_n": self.pred_frame_n,
            "tag": self.tag,
        }


class ComputeMetricsReq:
    def __init__(
        self,
        gt_video_infos: List[VideoInfo],
        pred_video_infos: List[VideoInfo],
        gt_video_anns: List[VideoAnnotation],
        pred_video_anns: List[VideoAnnotation],
        class_mapping: dict,
        tags_whitelist: List[str],
        obj_tags_whitelist: List[str],
        iou_threshold: float,
        segmentation_mode: bool,
        frame_from: Optional[int] = None,
        frame_to: Optional[int] = None,
    ) -> None:
        self.gt_video_infos = gt_video_infos
        self.pred_video_infos = pred_video_infos
        self.gt_video_anns = gt_video_anns
        self.pred_video_anns = pred_video_anns
        self.class_mapping = class_mapping
        self.tags_whitelist = tags_whitelist
        self.obj_tags_whitelist = obj_tags_whitelist
        self.iou_threshold = iou_threshold
        self.segmentation_mode = segmentation_mode
        self.frame_from = frame_from
        self.frame_to = frame_to


class ComputeMetricsVideoResult:
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.image_metrics: List[MetricValue] = []
        self.error_message = None

    def add(self, metric_value):
        self.image_metrics.append(metric_value)
        return self.image_metrics[-1]

    def to_json(self):
        if self.error_message is not None:
            return {"error": self.error_message}
        return [image_metric.to_json() for image_metric in self.image_metrics]


class ComputeMetricsResult:
    def __init__(self):
        self.video_metrics: List[ComputeMetricsVideoResult] = []
        self.overall_metrics: List[MetricValue] = []
        self.error_message = None

    def add_video_metrics(self, video_metrics):
        self.video_metrics.append(video_metrics)
        return self.video_metrics[-1]

    def add(self, metric: MetricValue):
        self.overall_metrics.append(metric)
        return self.overall_metrics[-1]

    def to_json(self):
        if self.error_message is not None:
            return {"error": self.error_message}
        return {
            "per_video": [
                {"video_name": video_metrics.video_name, "metrics": video_metrics.to_json()}
                for video_metrics in self.video_metrics
            ],
            "overall": [metric.to_json() for metric in self.overall_metrics],
        }


def _make_counters():
    return {TRUE_POSITIVE: 0, FALSE_POSITIVE: 0, FALSE_NEGATIVE: 0}


def _make_pixel_counters():
    return {INTERSECTION: 0, UNION: 0, ERROR_PIXELS: 0, TOTAL_PIXELS: 0}


def _fill_metric_value(
    metric_value,
    name,
    value,
    class_gt=None,
    gt_frame_n=None,
    pred_frame_n=None,
    tag=None,
):
    metric_value.value = value
    metric_value.metric_name = name
    if class_gt is not None:
        metric_value.class_gt = class_gt
    if gt_frame_n is not None:
        metric_value.gt_frame_n = gt_frame_n
    if pred_frame_n is not None:
        metric_value.pred_frame_n = pred_frame_n
    if tag is not None:
        metric_value.tag = tag


def _add_matching_metrics(
    dest,
    counters,
    metric_name_config,
    class_gt=None,
    gt_frame_n=None,
    pred_frame_n=None,
    tag=None,
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
            gt_frame_n=gt_frame_n,
            pred_frame_n=pred_frame_n,
            tag=tag,
        )

    return result_values


def _add_pixel_metrics(dest, counters, class_gt, gt_frame_n=None, pred_frame_n=None):
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
            gt_frame_n=gt_frame_n,
            pred_frame_n=pred_frame_n,
        )
    return result_values


def _maybe_add_average_metric(dest, metrics, metric_name, gt_frame_n=None, pred_frame_n=None):
    values = [m[metric_name] for m in metrics if metric_name in m]
    if len(values) > 0:
        avg_metric = np.mean(values).item()
        _fill_metric_value(
            dest.add(MetricValue()),
            metric_name,
            avg_metric,
            gt_frame_n=gt_frame_n,
            pred_frame_n=pred_frame_n,
        )
        return {metric_name: avg_metric}
    else:
        return {}


def add_tag_counts(
    dest_counters,
    dest_value_counters: dict,
    tags_gt,
    tags_pred,
    tags_whitelist,
    tp_key,
    fp_key,
    fn_key,
):
    effective_tags_gt = set((tag.name, tag.value) for tag in tags_gt if tag.name in tags_whitelist)
    effective_tags_pred = set(
        (tag.name, tag.value) for tag in tags_pred if tag.name in tags_whitelist
    )
    for name, value in effective_tags_pred - effective_tags_gt:
        dest_value_counters.setdefault((name, value), _make_counters())[fp_key] += 1
    for name, value in effective_tags_gt - effective_tags_pred:
        dest_value_counters.setdefault((name, value), _make_counters())[fn_key] += 1
    for name, value in effective_tags_gt & effective_tags_pred:
        dest_value_counters.setdefault((name, value), _make_counters())[tp_key] += 1

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


def is_segmentation(figure: VideoFigure, segmentation_mode: bool):
    if not segmentation_mode:
        return False
    if type(figure.geometry) in SEGMENTATION_GEOMETRIES:
        return True


def _tag_in_frame(tag: sly.VideoTag, frame_n):
    if tag.frame_range is None:
        return True
    return tag.frame_range[0] <= frame_n <= tag.frame_range[1]


def compute_metrics(
    request: ComputeMetricsReq, progress: Optional[CustomTqdm] = None
) -> Tuple[ComputeMetricsResult, List[List[Bitmap]]]:
    iou_threshold = request.iou_threshold
    tags_whitelist = set(request.tags_whitelist)
    obj_tags_whitelist = set(request.obj_tags_whitelist)
    gt_video_infos = request.gt_video_infos
    pred_video_infos = request.pred_video_infos
    gt_video_anns = request.gt_video_anns
    pred_video_anns = request.pred_video_anns
    class_mapping = request.class_mapping
    segmentation_mode = request.segmentation_mode
    frame_from = request.frame_from
    frame_to = request.frame_to

    difference_geometries = []
    result = ComputeMetricsResult()
    overall_class_matching_counters = {}
    overall_class_pixel_counters = {}
    overall_tag_counters = {}
    overall_tag_value_counters = {}
    overall_total_pixels = 0
    overall_total_pixel_error = 0

    try:
        for gt_video_info, pred_video_info, gt_video_ann, pred_video_ann in zip(
            gt_video_infos, pred_video_infos, gt_video_anns, pred_video_anns
        ):
            # if gt_video_info.name != pred_video_info.name:
            #     raise MetricsException(
            #         message="Ground truth video and Prediction video infos have different names."
            #     )
            vid_name = pred_video_info.name
            this_video_result = ComputeMetricsVideoResult(vid_name)
            this_difference_geometries = []
            frame_count = gt_video_info.frames_count
            video_shape = (gt_video_info.frame_height, gt_video_info.frame_width)
            if frame_count != pred_video_info.frames_count:
                raise MetricsException(
                    message="Ground truth video and Prediction video infos have different frames count."
                )
            if video_shape != (pred_video_info.frame_height, pred_video_info.frame_width):
                raise MetricsException(
                    message="Ground truth video and Prediction video infos have different frame sizes."
                )

            class_matching_counters = {class_gt: _make_counters() for class_gt in class_mapping}
            class_pixel_counters = {
                class_mapping[class_gt]: _make_pixel_counters() for class_gt in class_mapping
            }
            tag_counters = {
                tag_name: _make_counters()
                for tag_name in set(tags_whitelist) | set(obj_tags_whitelist)
            }
            tag_value_counters = {}
            total_pixel_error = 0
            total_pixels = 0

            add_tag_counts(
                tag_counters,
                tag_value_counters,
                gt_video_ann.tags,
                pred_video_ann.tags,
                tags_whitelist,
                tp_key=TRUE_POSITIVE,
                fp_key=FALSE_POSITIVE,
                fn_key=FALSE_NEGATIVE,
            )

            if frame_from is None:
                frame_from = 1
            if frame_to is None:
                frame_to = frame_count + 1
            for frame_n in range(frame_from, frame_to + 1):
                image_class_counters = {class_gt: _make_counters() for class_gt in class_mapping}
                image_pixel_counters = {
                    class_gt: _make_pixel_counters() for class_gt in class_mapping
                }
                image_tag_counters = {tag_name: _make_counters() for tag_name in tag_counters}
                image_tag_value_counters = {}

                image_class_overall_counters = _make_counters()
                image_tag_overall_counters = _make_counters()

                frame_errors_canvas = np.zeros(video_shape, dtype=np.bool)

                gt_frame_figures = [
                    fig for fig in gt_video_ann.figures if fig.frame_index + 1 == frame_n
                ]
                pred_frame_figures = [
                    fig for fig in pred_video_ann.figures if fig.frame_index + 1 == frame_n
                ]

                # non segmentation labels
                nonseg_gt_figures_idxs = [
                    idx
                    for idx, fig in enumerate(gt_frame_figures)
                    if not is_segmentation(fig, segmentation_mode)
                ]
                nonseg_pred_figures_idxs = [
                    idx
                    for idx, fig in enumerate(pred_frame_figures)
                    if not is_segmentation(fig, segmentation_mode)
                ]

                nonseg_gt_class_to_indices = {}
                for idx in nonseg_gt_figures_idxs:
                    fig = gt_frame_figures[idx]
                    nonseg_gt_class_to_indices.setdefault(
                        fig.video_object.obj_class.name, []
                    ).append(idx)
                nonseg_pred_class_to_indices = {}
                for idx in nonseg_pred_figures_idxs:
                    fig = pred_frame_figures[idx]
                    nonseg_pred_class_to_indices.setdefault(
                        fig.video_object.obj_class.name, []
                    ).append(idx)

                # segmentation labels
                seg_gt_figures_idxs = [
                    idx
                    for idx, figure in enumerate(gt_frame_figures)
                    if is_segmentation(figure, segmentation_mode)
                ]
                seg_pred_figure_idxs = [
                    idx
                    for idx, figure in enumerate(pred_frame_figures)
                    if is_segmentation(figure, segmentation_mode)
                ]

                seg_gt_class_to_indices = {}
                for idx in seg_gt_figures_idxs:
                    fig = gt_frame_figures[idx]
                    seg_gt_class_to_indices.setdefault(fig.video_object.obj_class.name, []).append(
                        idx
                    )
                seg_pred_class_to_indices = {}
                for idx in seg_pred_figure_idxs:
                    fig = pred_frame_figures[idx]
                    seg_pred_class_to_indices.setdefault(
                        fig.video_object.obj_class.name, []
                    ).append(idx)

                # segementation masks and canvas for segmentation labels
                (
                    seg_gt_effective_masks,
                    seg_gt_effective_canvas,
                ) = get_effective_nonoverlapping_masks(
                    [gt_frame_figures[idx].geometry for idx in seg_gt_figures_idxs],
                    img_size=video_shape,
                )
                gt_figure_idx_to_effective_mask_idx = {
                    idx: effective_idx for effective_idx, idx in enumerate(seg_gt_figures_idxs)
                }
                (
                    seg_pred_effective_masks,
                    seg_pred_effective_canvas,
                ) = get_effective_nonoverlapping_masks(
                    [pred_frame_figures[idx].geometry for idx in seg_pred_figure_idxs],
                    img_size=video_shape,
                )
                pred_figure_idx_to_effective_mask_idx = {
                    idx: effective_idx for effective_idx, idx in enumerate(seg_pred_figure_idxs)
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
                        [gt_frame_figures[idx].geometry for idx in gt_class_indices],
                        [pred_frame_figures[idx].geometry for idx in pred_class_indices],
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
                            image_tag_value_counters,
                            [
                                tag
                                for tag in gt_frame_figures[
                                    gt_class_indices[match.idx_1]
                                ].video_object.tags
                                if _tag_in_frame(tag, frame_n - 1)
                            ],
                            [
                                tag
                                for tag in pred_frame_figures[
                                    pred_class_indices[match.idx_2]
                                ].video_object.tags
                                if _tag_in_frame(tag, frame_n - 1)
                            ],
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )
                    for fn_label_idx in matching_results.unmatched_indices_1:
                        add_tag_counts(
                            image_tag_counters,
                            image_tag_value_counters,
                            [
                                tag
                                for tag in gt_frame_figures[
                                    gt_class_indices[fn_label_idx]
                                ].video_object.tags
                                if _tag_in_frame(tag, frame_n - 1)
                            ],
                            TagCollection(),
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )
                    for fp_label_idx in matching_results.unmatched_indices_2:
                        add_tag_counts(
                            image_tag_counters,
                            image_tag_value_counters,
                            TagCollection(),
                            [
                                tag
                                for tag in pred_frame_figures[
                                    pred_class_indices[fp_label_idx]
                                ].video_object.tags
                                if _tag_in_frame(tag, frame_n - 1)
                            ],
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )

                    # Iterating over matched labels
                    for gt_match, pred_match, _ in matching_results.matches:
                        gt_match_idx = gt_class_indices[gt_match]
                        pred_match_idx = pred_class_indices[pred_match]
                        gt_geometry = gt_frame_figures[gt_match_idx].geometry
                        pred_geometry = pred_frame_figures[pred_match_idx].geometry
                        gt_canvas = np.zeros(video_shape, dtype=np.bool)
                        gt_geometry.draw(gt_canvas, color=True)
                        pred_canvas = np.zeros(video_shape, dtype=np.bool)
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
                        frame_errors_canvas |= error_canvas

                    # Add errors to image errors canvas for unmatched labels
                    for gt_idx in matching_results.unmatched_indices_1:
                        gt_geometry = gt_frame_figures[gt_class_indices[gt_idx]].geometry
                        gt_canvas = np.zeros(video_shape, dtype=np.bool)
                        gt_geometry.draw(gt_canvas, color=True)
                        frame_errors_canvas |= gt_canvas
                    for pred_idx in matching_results.unmatched_indices_2:
                        pred_geometry = pred_frame_figures[pred_class_indices[pred_idx]].geometry
                        pred_canvas = np.zeros(video_shape, dtype=np.bool)
                        pred_geometry.draw(pred_canvas, color=True)
                        frame_errors_canvas |= pred_canvas

                    # Segmentation labels
                    gt_class_indices = seg_gt_class_to_indices.get(gt_class, [])
                    pred_class_indices = seg_pred_class_to_indices.get(pred_class, [])
                    seg_class_masks_gt = [
                        seg_gt_effective_masks[gt_figure_idx_to_effective_mask_idx[idx]]
                        for idx in gt_class_indices
                    ]
                    seg_class_masks_pred = [
                        seg_pred_effective_masks[pred_figure_idx_to_effective_mask_idx[idx]]
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
                            image_tag_value_counters,
                            [
                                tag
                                for tag in gt_frame_figures[
                                    gt_class_indices[match.idx_1]
                                ].video_object.tags
                                if _tag_in_frame(tag, frame_n - 1)
                            ],
                            [
                                tag
                                for tag in pred_frame_figures[
                                    pred_class_indices[match.idx_2]
                                ].video_object.tags
                                if _tag_in_frame(tag, frame_n - 1)
                            ],
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )

                    for fn_label_idx in matching_results.unmatched_indices_1:
                        add_tag_counts(
                            image_tag_counters,
                            image_tag_value_counters,
                            [
                                tag
                                for tag in gt_frame_figures[
                                    gt_class_indices[fn_label_idx]
                                ].video_object.tags
                                if _tag_in_frame(tag, frame_n - 1)
                            ],
                            TagCollection(),
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )

                    for fp_label_idx in matching_results.unmatched_indices_2:
                        add_tag_counts(
                            image_tag_counters,
                            image_tag_value_counters,
                            TagCollection(),
                            [
                                tag
                                for tag in pred_frame_figures[
                                    pred_class_indices[fp_label_idx]
                                ].video_object.tags
                                if _tag_in_frame(tag, frame_n - 1)
                            ],
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )

                    gt_canvas = np.isin(
                        seg_gt_effective_canvas,
                        [gt_figure_idx_to_effective_mask_idx[i] for i in gt_class_indices],
                    )
                    pred_canvas = np.isin(
                        seg_pred_effective_canvas,
                        [pred_figure_idx_to_effective_mask_idx[i] for i in pred_class_indices],
                    )
                    error_canvas = gt_canvas != pred_canvas

                    # Pixel counters
                    image_class_pixel_counters[INTERSECTION] += np.sum(
                        gt_canvas & pred_canvas
                    ).item()
                    image_class_pixel_counters[UNION] += np.sum(gt_canvas | pred_canvas).item()
                    image_class_pixel_counters[ERROR_PIXELS] += np.sum(error_canvas).item()
                    image_class_pixel_counters[TOTAL_PIXELS] += error_canvas.size

                    # Add nonrectangles differences to image errors canvas
                    frame_errors_canvas |= error_canvas

                    # Update image class counters
                    _sum_update_counters(class_pixel_counters[gt_class], image_class_pixel_counters)
                    _sum_update_counters(
                        class_matching_counters[gt_class], this_image_class_counters
                    )
                    _sum_update_counters(image_class_overall_counters, this_image_class_counters)

                # Update tag counters
                for tag_name, this_tag_counters in image_tag_counters.items():
                    _sum_update_counters(tag_counters[tag_name], this_tag_counters)
                    _sum_update_counters(image_tag_overall_counters, this_tag_counters)

                for (tag_name, tag_value), this_tag_counters in image_tag_value_counters.items():
                    tag_value_counters.setdefault((tag_name, tag_value), _make_counters())
                    _sum_update_counters(
                        tag_value_counters[(tag_name, tag_value)], this_tag_counters
                    )

                image_overall_score_components = {}

                # Object matching stats per image and class.
                for class_gt, this_class_counters in image_class_counters.items():
                    _add_matching_metrics(
                        this_video_result,
                        this_class_counters,
                        metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                        class_gt=class_gt,
                        gt_frame_n=frame_n,
                        pred_frame_n=frame_n,
                    )
                image_class_overall_metrics = _add_matching_metrics(
                    this_video_result,
                    image_class_overall_counters,
                    metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                    gt_frame_n=frame_n,
                    pred_frame_n=frame_n,
                )
                if MATCHES_F1 in image_class_overall_metrics:
                    image_overall_score_components.update(
                        {MATCHES_F1: image_class_overall_metrics[MATCHES_F1]}
                    )

                # Pixel accuracy metrics per image and class
                per_image_class_accuracy_metrics = {
                    class_gt: _add_pixel_metrics(
                        this_video_result,
                        image_class_pixel_counters,
                        class_gt,
                        gt_frame_n=frame_n,
                        pred_frame_n=frame_n,
                    )
                    for class_gt, image_class_pixel_counters in image_pixel_counters.items()
                }
                image_overall_score_components.update(
                    _maybe_add_average_metric(
                        this_video_result,
                        per_image_class_accuracy_metrics.values(),
                        IOU,
                        gt_frame_n=frame_n,
                        pred_frame_n=frame_n,
                    )
                )

                image_pixel_error = np.sum(frame_errors_canvas).item()
                num_image_pixels = frame_errors_canvas.size
                _fill_metric_value(
                    this_video_result.add(MetricValue()),
                    PIXEL_ACCURACY,
                    1.0 - image_pixel_error / num_image_pixels,
                    gt_frame_n=frame_n,
                    pred_frame_n=frame_n,
                )
                total_pixel_error += image_pixel_error
                total_pixels += num_image_pixels

                # Matching stats per image and tag.
                for tag_name, this_tag_counters in tag_counters.items():
                    _add_matching_metrics(
                        this_video_result,
                        this_tag_counters,
                        metric_name_config=_TAG_METRIC_NAMES,
                        gt_frame_n=frame_n,
                        pred_frame_n=frame_n,
                        tag=tag_name,
                    )
                for (tag_name, tag_value), this_tag_counters in tag_value_counters.items():
                    _add_matching_metrics(
                        this_video_result,
                        this_tag_counters,
                        metric_name_config=_TAG_METRIC_NAMES,
                        gt_frame_n=frame_n,
                        pred_frame_n=frame_n,
                        tag={"name": tag_name, "value": tag_value},
                    )
                image_tag_overall_metrics = _add_matching_metrics(
                    this_video_result,
                    image_tag_overall_counters,
                    metric_name_config=_TAG_METRIC_NAMES,
                    gt_frame_n=frame_n,
                    pred_frame_n=frame_n,
                )
                if TAGS_F1 in image_tag_overall_metrics:
                    image_overall_score_components.update(
                        {TAGS_F1: image_tag_overall_metrics[TAGS_F1]}
                    )

                if len(image_overall_score_components) > 0:
                    overall_score = np.mean(list(image_overall_score_components.values())).item()
                    _fill_metric_value(
                        this_video_result.add(MetricValue()),
                        OVERALL_SCORE,
                        overall_score,
                        gt_frame_n=frame_n,
                        pred_frame_n=frame_n,
                    )

                if np.any(frame_errors_canvas):
                    this_difference_geometries.append(Bitmap(frame_errors_canvas))
                else:
                    this_difference_geometries.append(None)

                if progress is not None:
                    progress.update(1)

            overall_score_components = {}

            # Overall metrics per class
            per_class_metrics = {
                class_gt: _add_matching_metrics(
                    this_video_result,
                    this_class_counters,
                    metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                    class_gt=class_gt,
                )
                for class_gt, this_class_counters in class_matching_counters.items()
            }
            for class_gt, this_class_counters in class_matching_counters.items():
                _sum_update_counters(
                    overall_class_matching_counters.setdefault(class_gt, _make_counters()),
                    this_class_counters,
                )
            overall_score_components.update(
                _maybe_add_average_metric(this_video_result, per_class_metrics.values(), MATCHES_F1)
            )

            # Per class pixel accuracy metrics.
            per_class_accuracy_metrics = {
                class_gt: _add_pixel_metrics(
                    this_video_result, image_class_pixel_counters, class_gt
                )
                for class_gt, image_class_pixel_counters in class_pixel_counters.items()
            }
            for class_gt, image_class_pixel_counters in class_pixel_counters.items():
                _sum_update_counters(
                    overall_class_pixel_counters.setdefault(class_gt, _make_pixel_counters()),
                    image_class_pixel_counters,
                )
            overall_score_components.update(
                _maybe_add_average_metric(
                    this_video_result, per_class_accuracy_metrics.values(), IOU
                )
            )
            overall_total_pixels += total_pixels
            overall_total_pixel_error += total_pixel_error
            if total_pixels > 0:
                _fill_metric_value(
                    this_video_result.add(MetricValue()),
                    PIXEL_ACCURACY,
                    1.0 - total_pixel_error / total_pixels,
                )

            per_tag_metrics = {
                tag_name: _add_matching_metrics(
                    this_video_result,
                    this_tag_counters,
                    metric_name_config=_TAG_METRIC_NAMES,
                    tag=tag_name,
                )
                for tag_name, this_tag_counters in tag_counters.items()
            }
            for tag_name, this_tag_counters in tag_counters.items():
                _sum_update_counters(
                    overall_tag_counters.setdefault(tag_name, _make_counters()), this_tag_counters
                )
            overall_score_components.update(
                _maybe_add_average_metric(this_video_result, per_tag_metrics.values(), TAGS_F1)
            )
            per_tag_value_metrics = {
                (tag_name, tag_value): _add_matching_metrics(
                    this_video_result,
                    this_tag_counters,
                    metric_name_config=_TAG_METRIC_NAMES,
                    tag={"name": tag_name, "value": tag_value},
                )
                for (tag_name, tag_value), this_tag_counters in tag_value_counters.items()
            }
            for (tag_name, tag_value), this_tag_counters in tag_value_counters.items():
                _sum_update_counters(
                    overall_tag_value_counters.setdefault(tag_name, _make_counters()),
                    this_tag_counters,
                )
            _maybe_add_average_metric(this_video_result, per_tag_value_metrics.values(), TAGS_F1)

            overall_score = 0
            if len(overall_score_components) > 0:
                overall_score = np.mean(list(overall_score_components.values())).item()
                _fill_metric_value(
                    this_video_result.add(MetricValue()), OVERALL_SCORE, overall_score
                )

            result.add_video_metrics(this_video_result)
            difference_geometries.append(this_difference_geometries)

        overall_score_components = {}
        # Overall metrics per class
        per_class_metrics = {
            class_gt: _add_matching_metrics(
                result,
                this_class_counters,
                metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                class_gt=class_gt,
            )
            for class_gt, this_class_counters in overall_class_matching_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(result, per_class_metrics.values(), MATCHES_F1)
        )

        # Per class pixel accuracy metrics.
        per_class_accuracy_metrics = {
            class_gt: _add_pixel_metrics(result, image_class_pixel_counters, class_gt)
            for class_gt, image_class_pixel_counters in overall_class_pixel_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(result, per_class_accuracy_metrics.values(), IOU)
        )

        if total_pixels > 0:
            _fill_metric_value(
                result.add(MetricValue()),
                PIXEL_ACCURACY,
                1.0 - overall_total_pixel_error / overall_total_pixels,
            )

        per_tag_metrics = {
            tag_name: _add_matching_metrics(
                result,
                this_tag_counters,
                metric_name_config=_TAG_METRIC_NAMES,
                tag=tag_name,
            )
            for tag_name, this_tag_counters in overall_tag_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(result, per_tag_metrics.values(), TAGS_F1)
        )
        per_tag_value_metrics = {
            (tag_name, tag_value): _add_matching_metrics(
                result,
                this_tag_counters,
                metric_name_config=_TAG_METRIC_NAMES,
                tag={"name": tag_name, "value": tag_value},
            )
            for (tag_name, tag_value), this_tag_counters in overall_tag_value_counters.items()
        }
        _maybe_add_average_metric(result, per_tag_value_metrics.values(), TAGS_F1)

        overall_score = 0
        if len(overall_score_components) > 0:
            overall_score = np.mean(list(overall_score_components.values())).item()
            _fill_metric_value(result.add(MetricValue()), OVERALL_SCORE, overall_score)

    except MetricsException as exc:
        # Reset the result to make sure there is no incomplete data there
        result = ComputeMetricsResult()
        result.error_message = exc.message
        difference_geometries = []
        if progress is not None:
            progress.update(gt_video_info.frames_count)
    except Exception as exc:
        result = ComputeMetricsResult()
        result.error_message = "Unexpected exception: {}".format(traceback.format_exc())
        difference_geometries = []
        if progress is not None:
            progress.update(gt_video_info.frames_count)
    return result, difference_geometries


@sly.timeit
def calculate_exam_report(
    gt_video_infos: List[VideoInfo],
    pred_video_infos: List[VideoInfo],
    gt_video_anns: List[VideoAnnotation],
    pred_video_anns: List[VideoAnnotation],
    class_mapping: dict,
    tags_whitelist: List[str],
    obj_tags_whitelist: List[str],
    iou_threshold: float,
    progress: Optional[CustomTqdm] = None,
    segmentation_mode: Optional[bool] = True,
) -> Tuple[List, List[Bitmap]]:
    request = ComputeMetricsReq(
        gt_video_infos=gt_video_infos,
        pred_video_infos=pred_video_infos,
        gt_video_anns=gt_video_anns,
        pred_video_anns=pred_video_anns,
        class_mapping=class_mapping,
        tags_whitelist=tags_whitelist,
        obj_tags_whitelist=obj_tags_whitelist,
        iou_threshold=iou_threshold,
        segmentation_mode=segmentation_mode,
    )
    result, diff_bitmaps = compute_metrics(request, progress)
    return result.to_json(), diff_bitmaps
