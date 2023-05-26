import json
import traceback
import numpy as np

import supervisely as sly

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag_collection import TagCollection
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.helpers import get_effective_nonoverlapping_masks
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
from supervisely.metric.matching import get_geometries_iou, match_indices_by_score
from supervisely.project.project_meta import ProjectMeta

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

PIXEL_ACCURACY = "pixel-accuracy"
ERROR_PIXELS = "error-pixels"
TOTAL_PIXELS = "total-pixels"

MATCHES_TRUE_POSITIVE = "matches-true-positive"
MATCHES_FALSE_POSITIVE = "matches-false-positive"
MATCHES_FALSE_NEGATIVE = "matches-false-negative"
MATCHES_PRECISION = "matches-precision"
MATCHES_RECALL = "matches-recall"
MATCHES_F1 = "matches-f1"

NUM_OBJECTS_GT = "num-objects-gt"
NUM_OBJECTS_PRED = "num-objects-pred"

TAGS_TRUE_POSITIVE = "tags-true-positive"
TAGS_FALSE_POSITIVE = "tags-false-positive"
TAGS_FALSE_NEGATIVE = "tags-false-negative"
TAGS_PRECISION = "tags-precision"
TAGS_RECALL = "tags-recall"
TAGS_F1 = "tags-f1"
TAGS_TOTAL_GT = "tags-total-gt"
TAGS_TOTAL_PRED = "tags-total-pred"

OVERALL_SCORE = "overall-score"

OVERALL_ERROR_CLASS = "error"


DEFAULT_IOU_THRESHOLD = 0.8

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


def get_iou_threshold(request):
    if request.iou_threshold != None:
        return request.iou_threshold
    return DEFAULT_IOU_THRESHOLD


def get_name(x):
    return x.name


class TutorialBackendException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


def _load_metas(api, project_gt_id, project_pred_id, class_matches):
    meta_gt = sly.ProjectMeta.from_json(api.project.get_meta(project_gt_id))
    meta_pred = sly.ProjectMeta.from_json(api.project.get_meta(project_pred_id))
    for class_match in class_matches:
        if not meta_gt.obj_classes.has_key(class_match.class_gt):
            raise TutorialBackendException(
                message=f"Requested object class {class_match.class_gt!r} is missing "
                f"from the ground truth project."
            )
        if not meta_pred.obj_classes.has_key(class_match.class_pred):
            raise TutorialBackendException(
                message=f"Requested object class {class_match.class_pred!r} is missing "
                f"from the evaluated project."
            )
    return meta_gt, meta_pred


def _load_image_infos(ds_info_gt, ds_info_pred, api):
    image_infos_gt = sorted(api.image.get_list(ds_info_gt.id), key=get_name)
    image_infos_pred = sorted(api.image.get_list(ds_info_pred.id), key=get_name)
    if len(image_infos_gt) != len(image_infos_pred):
        raise TutorialBackendException(
            message=f"Dataset {ds_info_gt.name!r} has different number of images in ground truth "
            f"and evaluation projects: {len(image_infos_gt)} vs {len(image_infos_pred)}."
        )
    for img_info_gt, img_info_pred in zip(image_infos_gt, image_infos_pred):
        if img_info_gt.hash != img_info_pred.hash:
            raise TutorialBackendException(
                message=f"Different image hashes found in ground truth and evaluation projects in "
                f"dataset {ds_info_gt.name!r}. {img_info_gt.name!r} with hash {img_info_gt.hash!r} "
                f"vs {img_info_pred.name!r} with hash {img_info_pred.hash!r}."
            )
    return image_infos_gt, image_infos_pred


def _load_annotation_jsons_batch(ds_id, image_infos, idx_batch, api):
    image_ids = [image_infos[i].id for i in idx_batch]
    ann_infos = api.annotation.download_batch(ds_id, image_ids)
    return [ann_info.annotation for ann_info in ann_infos]


def _fill_metric_value(
    metric_value,
    name,
    value,
    class_gt=None,
    image_gt_id=None,
    image_pred_id=None,
    image_dest_id=None,
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
    if image_dest_id is not None:
        metric_value.image_dest_id = image_dest_id
    if tag_name is not None:
        metric_value.tag_name = tag_name


def _add_matching_metrics(
    dest,
    counters,
    metric_name_config,
    class_gt=None,
    image_gt_id=None,
    image_pred_id=None,
    image_dest_id=None,
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
            / (
                2 * counters[TRUE_POSITIVE]
                + counters[FALSE_NEGATIVE]
                + counters[FALSE_POSITIVE]
            )
        )

    for out_name, val in result_values.items():
        _fill_metric_value(
            dest.add(MetricValue()),
            out_name,
            val,
            class_gt=class_gt,
            image_gt_id=image_gt_id,
            image_pred_id=image_pred_id,
            image_dest_id=image_dest_id,
            tag_name=tag_name,
        )

    return result_values


def _add_pixel_metrics(dest, counters, class_gt, image_gt_id=None, image_pred_id=None):
    result_values = dict()
    if counters[TOTAL_PIXELS] > 0:
        result_values[PIXEL_ACCURACY] = (
            1.0 - counters[ERROR_PIXELS] / counters[TOTAL_PIXELS]
        )
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


def _maybe_add_average_metric(
    dest, metrics, metric_name, image_gt_id=None, image_pred_id=None, image_dest_id=None
):
    values = [m[metric_name] for m in metrics if metric_name in m]
    if len(values) > 0:
        avg_metric = np.mean(values).item()
        _fill_metric_value(
            dest.add(MetricValue()),
            metric_name,
            avg_metric,
            image_gt_id=image_gt_id,
            image_pred_id=image_pred_id,
            image_dest_id=image_dest_id,
        )
        return {metric_name: avg_metric}
    else:
        return {}


def add_tag_counts(
    dest_counters, tags_gt, tags_pred, tags_whitelist, tp_key, fp_key, fn_key
):
    effective_tags_gt = set(
        (tag.name, tag.value) for tag in tags_gt if tag.name in tags_whitelist
    )
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


def _maybe_label(obj_class, data):
    if np.any(data):
        return [Label(obj_class=obj_class, geometry=Bitmap(data=data))]
    else:
        return []


def _make_counters():
    return {TRUE_POSITIVE: 0, FALSE_POSITIVE: 0, FALSE_NEGATIVE: 0}


def _make_pixel_counters():
    return {INTERSECTION: 0, UNION: 0, ERROR_PIXELS: 0, TOTAL_PIXELS: 0}


class ClassMatch:
    def __init__(self, class_gt, class_pred):
        self.class_gt = class_gt
        self.class_pred = class_pred


class ComputeMetricsReq:
    def __init__(
        self,
        server_address,
        api_token,
        project_gt_id,
        dataset_gt_id,
        project_pred_id,
        dataset_pred_id,
        project_dest_id,
        dataset_dest_id,
        class_matches,
        tags_whitelist,
        obj_tags_whitelist,
        iou_threshold,
    ):
        self.server_address = server_address
        self.api_token = api_token
        self.project_gt_id = project_gt_id
        self.dataset_gt_id = dataset_gt_id
        self.project_pred_id = project_pred_id
        self.dataset_pred_id = dataset_pred_id
        self.project_dest_id = project_dest_id
        self.dataset_dest_id = dataset_dest_id
        self.class_matches = class_matches
        self.tags_whitelist = tags_whitelist
        self.obj_tags_whitelist = obj_tags_whitelist
        self.iou_threshold = iou_threshold


class ComputeMetricsResp:
    def __init__(self):
        self.image_metrics = []

    def add(self, metric_value):
        self.image_metrics.append(metric_value)
        return self.image_metrics[-1]

    def to_json(self):
        return [image_metric.to_json() for image_metric in self.image_metrics]


class MetricValue:
    def __init__(self):
        self.value = 0
        self.metric_name = ""
        self.class_gt = ""
        self.image_gt_id = 0
        self.image_pred_id = 0
        self.image_dest_id = 0
        self.tag_name = ""

    def to_json(self):
        return {
            "value": self.value,
            "metric_name": self.metric_name,
            "class_gt": self.class_gt,
            "image_gt_id": self.image_gt_id,
            "image_pred_id": self.image_pred_id,
            "image_dest_id": self.image_dest_id,
            "tag_name": self.tag_name,
        }


def ComputeMetrics(request: ComputeMetricsReq):
    # Initialize API.
    response = ComputeMetricsResp()
    api = sly.Api(server_address=request.server_address, token=request.api_token)

    iou_threshold = get_iou_threshold(request)
    tags_whitelist = set(request.tags_whitelist)
    obj_tags_whitelist = set(request.obj_tags_whitelist)

    try:
        # Open both projects.
        # Make sure metas have the requested classes from the mapping.
        meta_gt, meta_pred = _load_metas(
            api,
            request.project_gt_id,
            request.project_pred_id,
            request.class_matches,
        )

        # Make sure dataset names and image hashes match.
        ds_info_gt = api.dataset.get_info_by_id(request.dataset_gt_id)
        ds_info_pred = api.dataset.get_info_by_id(request.dataset_pred_id)
        ds_info_dest = api.dataset.get_info_by_id(request.dataset_dest_id)
        image_infos_gt, image_infos_pred = _load_image_infos(
            ds_info_gt, ds_info_pred, api
        )
        _, image_infos_dest = _load_image_infos(ds_info_gt, ds_info_dest, api)

        class_mapping = {
            class_match.class_gt: class_match.class_pred
            for class_match in request.class_matches
        }

        overall_error_class = ObjClass(
            name=OVERALL_ERROR_CLASS, geometry_type=Bitmap, color=[255, 0, 0]
        )
        dest_meta = ProjectMeta(obj_classes=ObjClassCollection([overall_error_class]))
        api.project.update_meta(request.project_dest_id, dest_meta.to_json())

        class_matching_counters = {
            class_gt: _make_counters() for class_gt in class_mapping
        }
        class_pixel_counters = {
            class_gt: _make_pixel_counters() for class_gt in class_mapping
        }
        tag_counters = {
            tag_name: _make_counters()
            for tag_name in set(tags_whitelist) | set(obj_tags_whitelist)
        }
        total_pixel_error = 0
        total_pixels = 0

        # In batches, download annotations for the ground truth and predictions.
        for idx_batch in sly.batched(range(len(image_infos_gt))):
            image_infos_gt_batch = [image_infos_gt[idx] for idx in idx_batch]
            image_infos_pred_batch = [image_infos_pred[idx] for idx in idx_batch]
            image_infos_dest_batch = [image_infos_dest[idx] for idx in idx_batch]

            ann_jsons_gt = _load_annotation_jsons_batch(
                ds_info_gt.id, image_infos_gt, idx_batch, api
            )
            ann_jsons_pred = _load_annotation_jsons_batch(
                ds_info_pred.id, image_infos_pred, idx_batch, api
            )
            if len(ann_jsons_gt) != len(ann_jsons_pred):
                raise TutorialBackendException(
                    message=f"Downloaded annotation batches have different lengths in ground truth and "
                    f"evaluation projects in dataset {ds_info_gt.name!r}."
                )

            dest_anns_batch = []

            # Pass the annotations in pairs through the metric computers.
            for (
                image_info_gt,
                image_info_pred,
                image_info_dest,
                ann_json_gt,
                ann_json_pred,
            ) in zip(
                image_infos_gt_batch,
                image_infos_pred_batch,
                image_infos_dest_batch,
                ann_jsons_gt,
                ann_jsons_pred,
            ):
                ann_gt = sly.Annotation.from_json(ann_json_gt, meta_gt)
                ann_pred = sly.Annotation.from_json(ann_json_pred, meta_pred)

                image_class_counters = {
                    class_gt: _make_counters() for class_gt in class_mapping
                }
                image_pixel_counters = {
                    class_gt: _make_pixel_counters() for class_gt in class_mapping
                }
                image_tag_counters = {
                    tag_name: _make_counters() for tag_name in tag_counters
                }

                image_class_overall_counters = _make_counters()
                image_tag_overall_counters = _make_counters()

                add_tag_counts(
                    image_tag_counters,
                    ann_gt.img_tags,
                    ann_pred.img_tags,
                    tags_whitelist,
                    tp_key=TRUE_POSITIVE,
                    fp_key=FALSE_POSITIVE,
                    fn_key=FALSE_NEGATIVE,
                )

                image_canvas_errors = np.zeros(ann_gt.img_size, dtype=np.bool)

                # Render labels and get effective masks.
                (
                    effective_masks_gt,
                    effective_canvas_gt,
                ) = get_effective_nonoverlapping_masks(
                    [label.geometry for label in ann_gt.labels],
                    img_size=ann_gt.img_size,
                )
                (
                    effective_masks_pred,
                    effective_canvas_pred,
                ) = get_effective_nonoverlapping_masks(
                    [label.geometry for label in ann_pred.labels],
                    img_size=ann_pred.img_size,
                )

                class_to_indices_gt = {}
                for idx, label in enumerate(ann_gt.labels):
                    class_to_indices_gt.setdefault(label.obj_class.name, []).append(idx)
                # TODO refactor with above
                class_to_indices_pred = {}
                for idx, label in enumerate(ann_pred.labels):
                    class_to_indices_pred.setdefault(label.obj_class.name, []).append(
                        idx
                    )

                for class_gt, class_pred in class_mapping.items():
                    this_class_indices_gt = class_to_indices_gt.get(class_gt, [])
                    this_class_indices_pred = class_to_indices_pred.get(class_pred, [])

                    class_canvas_gt = np.isin(
                        effective_canvas_gt, this_class_indices_gt
                    )
                    class_canvas_pred = np.isin(
                        effective_canvas_pred, this_class_indices_pred
                    )
                    class_canvas_err = class_canvas_gt != class_canvas_pred

                    image_class_pixel_counters = image_pixel_counters[class_gt]
                    image_class_pixel_counters[INTERSECTION] = np.sum(
                        class_canvas_gt & class_canvas_pred
                    ).item()
                    image_class_pixel_counters[UNION] = np.sum(
                        class_canvas_gt | class_canvas_pred
                    ).item()
                    image_class_pixel_counters[ERROR_PIXELS] = np.sum(
                        class_canvas_err
                    ).item()
                    image_class_pixel_counters[TOTAL_PIXELS] = class_canvas_err.size

                    _sum_update_counters(
                        class_pixel_counters[class_gt], image_class_pixel_counters
                    )

                    image_canvas_errors |= class_canvas_err

                    class_masks_gt = [
                        effective_masks_gt[idx] for idx in this_class_indices_gt
                    ]
                    class_masks_pred = [
                        effective_masks_pred[idx] for idx in this_class_indices_pred
                    ]

                    # TODO: IOU threshold to config
                    class_matched_indices = match_indices_by_score(
                        class_masks_gt,
                        class_masks_pred,
                        iou_threshold,
                        safe_get_geometries_iou,
                    )

                    # Object matching stats.
                    this_image_class_counters = image_class_counters[class_gt]
                    this_image_class_counters[TRUE_POSITIVE] = len(
                        class_matched_indices.matches
                    )
                    this_image_class_counters[FALSE_NEGATIVE] = len(
                        class_matched_indices.unmatched_indices_1
                    )
                    this_image_class_counters[FALSE_POSITIVE] = len(
                        class_matched_indices.unmatched_indices_2
                    )
                    _sum_update_counters(
                        class_matching_counters[class_gt], this_image_class_counters
                    )
                    _sum_update_counters(
                        image_class_overall_counters, this_image_class_counters
                    )

                    # Tags matching stats.
                    for match in class_matched_indices.matches:
                        add_tag_counts(
                            image_tag_counters,
                            ann_gt.labels[this_class_indices_gt[match.idx_1]].tags,
                            ann_pred.labels[this_class_indices_pred[match.idx_2]].tags,
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )

                    for fn_label_idx in class_matched_indices.unmatched_indices_1:
                        add_tag_counts(
                            image_tag_counters,
                            ann_gt.labels[this_class_indices_gt[fn_label_idx]].tags,
                            TagCollection(),
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )

                    for fp_label_idx in class_matched_indices.unmatched_indices_2:
                        add_tag_counts(
                            image_tag_counters,
                            TagCollection(),
                            ann_pred.labels[this_class_indices_pred[fp_label_idx]].tags,
                            obj_tags_whitelist,
                            tp_key=TRUE_POSITIVE,
                            fp_key=FALSE_POSITIVE,
                            fn_key=FALSE_NEGATIVE,
                        )

                for tag_name, this_tag_counters in image_tag_counters.items():
                    _sum_update_counters(tag_counters[tag_name], this_tag_counters)
                    _sum_update_counters(image_tag_overall_counters, this_tag_counters)

                image_overall_score_components = {}

                # Object matching stats per image and class.
                for class_gt, this_class_counters in image_class_counters.items():
                    _add_matching_metrics(
                        response,
                        this_class_counters,
                        metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                        class_gt=class_gt,
                        image_gt_id=image_info_gt.id,
                        image_pred_id=image_info_pred.id,
                        image_dest_id=image_info_dest.id,
                    )
                image_class_overall_metrics = _add_matching_metrics(
                    response,
                    image_class_overall_counters,
                    metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                    image_gt_id=image_info_gt.id,
                    image_pred_id=image_info_pred.id,
                    image_dest_id=image_info_dest.id,
                )
                if MATCHES_F1 in image_class_overall_metrics:
                    image_overall_score_components.update(
                        {MATCHES_F1: image_class_overall_metrics[MATCHES_F1]}
                    )

                # Pixel accuracy metrics per image and class
                per_image_class_accuracy_metrics = {
                    class_gt: _add_pixel_metrics(
                        response,
                        image_class_pixel_counters,
                        class_gt,
                        image_gt_id=image_info_gt.id,
                        image_pred_id=image_info_pred.id,
                    )
                    for class_gt, image_class_pixel_counters in image_pixel_counters.items()
                }
                image_overall_score_components.update(
                    _maybe_add_average_metric(
                        response,
                        per_image_class_accuracy_metrics.values(),
                        IOU,
                        image_gt_id=image_info_gt.id,
                        image_pred_id=image_info_pred.id,
                        image_dest_id=image_info_dest.id,
                    )
                )

                image_pixel_error = np.sum(image_canvas_errors).sum().item()
                num_image_pixels = image_canvas_errors.size
                _fill_metric_value(
                    response.add(MetricValue()),
                    PIXEL_ACCURACY,
                    1.0 - image_pixel_error / num_image_pixels,
                    image_gt_id=image_info_gt.id,
                    image_pred_id=image_info_pred.id,
                    image_dest_id=image_info_dest.id,
                )
                total_pixel_error += image_pixel_error
                total_pixels += num_image_pixels

                # Matching stats per image and tag.
                for tag_name, this_tag_counters in tag_counters.items():
                    _add_matching_metrics(
                        response,
                        this_tag_counters,
                        metric_name_config=_TAG_METRIC_NAMES,
                        image_gt_id=image_info_gt.id,
                        image_pred_id=image_info_pred.id,
                        image_dest_id=image_info_dest.id,
                        tag_name=tag_name,
                    )
                image_tag_overall_metrics = _add_matching_metrics(
                    response,
                    image_tag_overall_counters,
                    metric_name_config=_TAG_METRIC_NAMES,
                    image_gt_id=image_info_gt.id,
                    image_pred_id=image_info_pred.id,
                    image_dest_id=image_info_dest.id,
                )
                if TAGS_F1 in image_tag_overall_metrics:
                    image_overall_score_components.update(
                        {TAGS_F1: image_tag_overall_metrics[TAGS_F1]}
                    )

                if len(image_overall_score_components) > 0:
                    overall_score = np.mean(
                        list(image_overall_score_components.values())
                    ).item()
                    _fill_metric_value(
                        response.add(MetricValue()),
                        OVERALL_SCORE,
                        overall_score,
                        image_gt_id=image_info_gt.id,
                        image_pred_id=image_info_pred.id,
                        image_dest_id=image_info_dest.id,
                    )

                dest_anns_batch.append(
                    Annotation(
                        img_size=ann_gt.img_size,
                        labels=_maybe_label(overall_error_class, image_canvas_errors),
                    )
                )

            # Upload annotation batch with difference renders.
            api.annotation.upload_anns(
                img_ids=[image_infos_dest[idx].id for idx in idx_batch],
                anns=dest_anns_batch,
            )

        overall_score_components = {}

        # Overall metrics per class
        per_class_metrics = {
            class_gt: _add_matching_metrics(
                response,
                this_class_counters,
                metric_name_config=_OBJ_MATCHES_METRIC_NAMES,
                class_gt=class_gt,
            )
            for class_gt, this_class_counters in class_matching_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(response, per_class_metrics.values(), MATCHES_F1)
        )

        # Per class pixel accuracy metrics.
        per_class_accuracy_metrics = {
            class_gt: _add_pixel_metrics(response, image_class_pixel_counters, class_gt)
            for class_gt, image_class_pixel_counters in class_pixel_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(
                response, per_class_accuracy_metrics.values(), IOU
            )
        )
        if total_pixels > 0:
            _fill_metric_value(
                response.add(MetricValue()),
                PIXEL_ACCURACY,
                1.0 - total_pixel_error / total_pixels,
            )

        per_tag_metrics = {
            tag_name: _add_matching_metrics(
                response,
                this_tag_counters,
                metric_name_config=_TAG_METRIC_NAMES,
                tag_name=tag_name,
            )
            for tag_name, this_tag_counters in tag_counters.items()
        }
        overall_score_components.update(
            _maybe_add_average_metric(response, per_tag_metrics.values(), TAGS_F1)
        )

        if len(overall_score_components) > 0:
            overall_score = np.mean(list(overall_score_components.values())).item()
            _fill_metric_value(
                response.add(MetricValue()), OVERALL_SCORE, overall_score
            )

    except TutorialBackendException as exc:
        # Reset the response to make sure there is no incomplete data there
        response = ComputeMetricsResp()
        response.error_message = exc.message
    except Exception as exc:
        response = ComputeMetricsResp()
        response.error_message = "Unexpected exception: {}".format(
            traceback.format_exc()
        )

    return response


def calculate_exam_report(
    server_address,
    api_token,
    project_gt_id,
    dataset_gt_id,
    project_pred_id,
    dataset_pred_id,
    project_dest_id,
    dataset_dest_id,
    class_matches,
    tags_whitelist,
    obj_tags_whitelist,
    iou_threshold,
):
    class_matches = [
        ClassMatch(class_gt=cm["class_gt"], class_pred=cm["class_pred"])
        for cm in class_matches
    ]
    request = ComputeMetricsReq(
        server_address=server_address,
        api_token=api_token,
        project_gt_id=project_gt_id,
        dataset_gt_id=dataset_gt_id,
        project_pred_id=project_pred_id,
        dataset_pred_id=dataset_pred_id,
        project_dest_id=project_dest_id,
        dataset_dest_id=dataset_dest_id,
        class_matches=class_matches,
        tags_whitelist=tags_whitelist,
        obj_tags_whitelist=obj_tags_whitelist,
        iou_threshold=iou_threshold
    )
    result = ComputeMetrics(request)
    return result.to_json()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

    server_address = sly.env.server_address()
    api_token = sly.env.api_token()
    project_gt_id = int(os.environ.get("PROJECT_GT_ID"))
    dataset_gt_id = int(os.environ.get("DATASET_GT_ID"))
    project_pred_id = int(os.environ.get("PROJECT_PRED_ID"))
    dataset_pred_id = int(os.environ.get("DATASET_PRED_ID"))
    project_dest_id = int(os.environ.get("PROJECT_DEST_ID"))
    dataset_dest_id = int(os.environ.get("DATASET_DEST_ID"))
    class_matches = [
        ClassMatch(class_gt="kiwi", class_pred="kiwi"),
        ClassMatch(class_gt="lemon", class_pred="lemon"),
    ]
    tags_whitelist = []
    obj_tags_whitelist = []

    result = calculate_exam_report(
        server_address,
        api_token,
        project_gt_id,
        dataset_gt_id,
        project_pred_id,
        dataset_pred_id,
        project_dest_id,
        dataset_dest_id,
        class_matches,
        tags_whitelist,
        obj_tags_whitelist,
    )

    with open("result.json", "w") as f:
        f.write(json.dumps(result.to_json(), indent=2))
