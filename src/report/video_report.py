import copy
import json
from pathlib import Path
import time
from typing import List

import supervisely as sly
from supervisely import VideoAnnotation
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import (
    Container,
    Text,
    Card,
    Table,
    GridGallery,
    Timeline,
    Field,
    InputNumber,
    Button,
    Switch,
    Flexbox,
    Select,
    Progress,
)
from supervisely.app import DataJson
from supervisely.imaging.image import write as write_image

from src.report.report import (
    return_button,
    get_diff_dataset,
    get_diff_project,
    get_or_create_diff_project,
    error_notification,
    error_in_report,
    assigned_to,
    exam_passmark,
    exam_score,
    status,
    benchmark_project_thumbnail,
    overall_stats,
)
from src.report.video_metrics import calculate_exam_report
import src.globals as g
from src.exam import Exam


GOOD_TIMELINE_COLOR = "#91B974"
ERROR_TIMELINE_COLOR = "#FF6458"
EMPTY_TIMELINE_COLOR = "#d1dbe5"

current_report = None
selected_video = None
gt_vids: List[VideoInfo] = []
pred_vids: List[VideoInfo] = []
diff_vids: List[VideoInfo] = []
gt_annotations: List[VideoAnnotation] = []
pred_annotations: List[VideoAnnotation] = []
diff_annotations: List[VideoAnnotation] = []
current_frame_range = [1, 1]


# overall_score = Text("")
# overall_stats = Card(title="Overall Score", content=overall_score)

obj_count_per_class_table_columns = [
    "NAME",
    "GT Objects",
    "Labeled Objects",
    "Recall(Matched objects)",
    "Precision",
    "F-measure",
]
obj_count_per_class_table = Table(columns=obj_count_per_class_table_columns)
obj_count_per_class_last = Text()
obj_count_per_class = Card(
    title="OBJECTS COUNT PER CLASS",
    content=Container(widgets=[obj_count_per_class_table, obj_count_per_class_last], gap=5),
)

geometry_quality_table_columns = ["NAME", "Pixel Accuracy", "IOU"]
geometry_quality_table = Table(columns=geometry_quality_table_columns)
geometry_quality_last = Text()
geometry_quality = Card(
    title="GEOMETRY QUALITY",
    content=Container(widgets=[geometry_quality_table, geometry_quality_last], gap=5),
)

tags_stat_table_columns = [
    "NAME",
    "GT Tags",
    "Labeled Tags",
    "Precision",
    "Recall",
    "F-measure",
]
tags_stat_table = Table(columns=tags_stat_table_columns)
tags_stat_last = Text()

tags_values_stat_table_columns = [
    "Tag Value",
    "GT Tags",
    "Labeled Tags",
    "Precision",
    "Recall",
    "F-measure",
]
tags_values_stat_table = Table(columns=tags_values_stat_table_columns)
tags_values_stat_card = Card(
    title="Tags values stats",
    description="Click on a tags table row to see stats for each tag value",
    content=tags_values_stat_table,
    collapsable=True,
)
tags_values_stat_card.collapse()


@tags_stat_table.click
def tags_stat_table_click_cb(datapoint):
    tag_name = datapoint.row["NAME"]
    tags_values_stat_table.read_json(
        {
            "columns": tags_values_stat_table_columns,
            "data": [row for row in get_tags_values_stat_table_rows(current_report, tag_name)],
        }
    )
    tags_values_stat_card.uncollapse()


tags_stat = Card(
    title="TAGS",
    content=Container(widgets=[tags_stat_table, tags_stat_last, tags_values_stat_card], gap=5),
)

report_per_image_video_select = Select([])
report_per_image_table_columns = [
    "FRAME #",
    "Objects Score",
    "Objects Missing",
    "Objects False Positive",
    "Tags Score",
    "Tags Missing",
    "Tags False Positive",
    "Geometry Score",
    "Overall Score",
]
report_per_image_table = Table(columns=report_per_image_table_columns)
report_per_image_table.click(lambda x: x)
timeline = Timeline(1, [], [])
timeline_select_frame = InputNumber(min=1, max=1, value=1)
timeline_filters_widgets = {
    "objects_score": {
        "switch": Switch(),
        "input": InputNumber(min=0, max=100, value=80, step=0.1, precision=1),
    },
    "objects_fp": {"switch": Switch(), "input": InputNumber(min=0, value=0)},
    "objects_fn": {"switch": Switch(), "input": InputNumber(min=0, value=0)},
    "tags_score": {
        "switch": Switch(),
        "input": InputNumber(min=0, max=100, value=80, step=0.1, precision=1),
    },
    "tags_fp": {"switch": Switch(), "input": InputNumber(min=0, value=0)},
    "tags_fn": {"switch": Switch(), "input": InputNumber(min=0, value=0)},
    "geometry_score": {
        "switch": Switch(),
        "input": InputNumber(min=0, max=100, value=80, step=0.1, precision=1),
    },
    "overall_score": {
        "switch": Switch(),
        "input": InputNumber(min=0, max=100, value=80, step=0.1, precision=1),
    },
}
timeline_filters_apply_btn = Button("Apply")
timeline_filters = Flexbox(
    widgets=[
        Container(
            [
                Field(
                    title="Objects score",
                    description="If frame Objects Score is less than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["objects_score"]["switch"],
                            timeline_filters_widgets["objects_score"]["input"],
                        ],
                    ),
                ),
                Field(
                    title="Objects FN",
                    description="If Missing Objects number is more than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["objects_fn"]["switch"],
                            timeline_filters_widgets["objects_fn"]["input"],
                        ]
                    ),
                ),
                Field(
                    title="Objects FP",
                    description="If False Positive Objects number is more than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["objects_fp"]["switch"],
                            timeline_filters_widgets["objects_fp"]["input"],
                        ]
                    ),
                ),
            ]
        ),
        Container(
            [
                Field(
                    title="Tags score",
                    description="If frame Tags Score is less than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["tags_score"]["switch"],
                            timeline_filters_widgets["tags_score"]["input"],
                        ]
                    ),
                ),
                Field(
                    title="Tags FP",
                    description="If Missing Tags number is more than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["tags_fp"]["switch"],
                            timeline_filters_widgets["tags_fp"]["input"],
                        ]
                    ),
                ),
                Field(
                    title="Tags FN",
                    description="If False Positive Tags number is more than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["tags_fn"]["switch"],
                            timeline_filters_widgets["tags_fn"]["input"],
                        ]
                    ),
                ),
            ]
        ),
        Container(
            [
                Field(
                    title="Geometry score",
                    description="If frame Geometry Score is less than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["geometry_score"]["switch"],
                            timeline_filters_widgets["geometry_score"]["input"],
                        ]
                    ),
                ),
                Field(
                    title="Overall score",
                    description="If frame Overall Score is less than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["overall_score"]["switch"],
                            timeline_filters_widgets["overall_score"]["input"],
                        ]
                    ),
                ),
            ]
        ),
    ],
    gap=30,
)


@report_per_image_video_select.value_changed
def video_changed(value):
    if value is None:
        return
    vid = None
    for vid in pred_vids:
        if vid.name == value:
            break
    if vid is None:
        return
    timeline_select_frame.min = 0
    timeline_select_frame.max = vid.frames_count
    timeline_select_frame.value = timeline_select_frame.min
    global selected_video
    selected_video = value


@timeline_select_frame.value_changed
def timeline_select_frame_cb(frame_n):
    global current_frame_range
    pointer = frame_n - current_frame_range[0]
    timeline.set_pointer(pointer)
    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [get_report_per_image_row(current_report, selected_video, frame_n)],
        }
    )
    show_images(frame_n)


def get_timeline_filters():
    filters = {}
    for metric_name in [
        "objects_score",
        "objects_fp",
        "objects_fn",
        "tags_score",
        "tags_fp",
        "tags_fn",
        "geometry_score",
        "overall_score",
    ]:
        if timeline_filters_widgets[metric_name]["switch"].is_switched():
            filters[metric_name] = timeline_filters_widgets[metric_name]["input"].get_value()
        else:
            filters[metric_name] = None
    return filters


def report_to_dict(report):
    report_dict = {"per_video": {}}
    for video_metrics in report.get("per_video", []):
        video_name = video_metrics["video_name"]
        metrics = video_metrics["metrics"]
        d = {}
        for metric in metrics:
            if metric["metric_name"] not in d:
                d[metric["metric_name"]] = {}
            if metric["gt_frame_n"] not in d[metric["metric_name"]]:
                d[metric["metric_name"]][metric["gt_frame_n"]] = {
                    "pred_frame_n": metric["pred_frame_n"]
                }
            d[metric["metric_name"]][metric["gt_frame_n"]][
                (
                    metric["class_gt"],
                    metric["tag"] if isinstance(metric["tag"], str) else json.dumps(metric["tag"]),
                )
            ] = metric["value"]
        report_dict["per_video"][video_name] = d
    d = {}
    for metric in report.get("overall", []):
        if metric["metric_name"] not in d:
            d[metric["metric_name"]] = {}
        if metric["gt_frame_n"] not in d[metric["metric_name"]]:
            d[metric["metric_name"]][metric["gt_frame_n"]] = {
                "pred_frame_n": metric["pred_frame_n"]
            }
        d[metric["metric_name"]][metric["gt_frame_n"]][
            (
                metric["class_gt"],
                metric["tag"] if isinstance(metric["tag"], str) else json.dumps(metric["tag"]),
            )
        ] = metric["value"]
    report_dict["overall"] = d
    return report_dict


def download_frame(video_id, frame_n):
    return g.api.video.frame.download_np(video_id, frame_n)


def save_img(np, filename):
    write_image(str(Path(g.TEMP_DATA_PATH).joinpath(filename)), np)


def get_vid_infos(dataset: sly.DatasetInfo):
    return sorted(g.api.video.get_list(dataset.id), key=lambda x: x.id)


def create_diff_dataset(diff_project: sly.ProjectInfo, attempt_dataset: sly.DatasetInfo):
    diff_dataset = g.api.dataset.create(diff_project.id, attempt_dataset.name)
    vids = g.api.video.get_list(attempt_dataset.id)
    vid_ids, vid_names = zip(*[(img.id, img.name) for img in vids])
    g.api.video.upload_ids(diff_dataset.id, vid_names, vid_ids)

    return diff_dataset


def get_or_create_diff_dataset(diff_project: sly.ProjectInfo, attempt_dataset: sly.DatasetInfo):
    diff_dataset = get_diff_dataset(diff_project)
    if diff_dataset is None:
        diff_dataset = create_diff_dataset(diff_project, attempt_dataset)
    return diff_dataset


def show_images(frame_n):
    global report_per_image_images

    report_per_image_images.loading = True
    report_per_image_images.clean_up()

    vid_name = report_per_image_video_select.get_value()
    pred_video_info = None
    for i, pred_vid in enumerate(pred_vids):
        if pred_vid.name == vid_name:
            gt_video_info = gt_vids[i]
            pred_video_info = pred_vid
            diff_video_info = diff_vids[i]
            gt_annotation = gt_annotations[i]
            pred_annotation = pred_annotations[i]
            diff_annotation = diff_annotations[i]
            break
    if pred_video_info is None:
        raise RuntimeError("Video not found")

    # gt image
    frame_np = download_frame(gt_video_info.id, frame_n - 1)
    save_img(frame_np, "gt.jpg")

    # pred image
    if pred_video_info.id != gt_video_info.id:
        frame_np = download_frame(pred_video_info.id, frame_n - 1)
    save_img(frame_np, "pred.jpg")

    # gt annotation
    frame_shape = (gt_video_info.frame_height, gt_video_info.frame_width)
    labels = [
        sly.Label(geometry=fig.geometry, obj_class=fig.video_object.obj_class)
        for fig in gt_annotation.figures
        if fig.frame_index == frame_n - 1
    ]
    gt_ann = sly.Annotation(frame_shape, labels=labels)

    # pred annotation
    frame_shape = (pred_video_info.frame_height, pred_video_info.frame_width)
    labels = [
        sly.Label(geometry=fig.geometry, obj_class=fig.video_object.obj_class)
        for fig in pred_annotation.figures
        if fig.frame_index == frame_n - 1
    ]
    pred_ann = sly.Annotation(frame_shape, labels=labels)

    # diff annotation
    frame_shape = (gt_video_info.frame_height, gt_video_info.frame_width)
    try:
        frame = diff_annotation.frames.get(frame_n - current_frame_range[0])
        labels = [
            sly.Label(
                fig.geometry,
                sly.ObjClass("difference", sly.Bitmap, (255, 0, 0)),
            )
            for fig in frame.figures
        ]
    except Exception:
        labels = []
    diff_ann = sly.Annotation(
        img_size=frame_shape,
        labels=labels,
    )
    report_per_image_images.clean_up()
    report_per_image_images.append(
        f"./static/gt.jpg?{time.time()}", gt_ann, title=f"GT: {vid_name}", column_index=0
    )
    report_per_image_images.append(
        f"./static/pred.jpg?{time.time()}",
        pred_ann,
        title=f"Pred: {vid_name}",
        column_index=1,
    )
    report_per_image_images.append(
        f"./static/gt.jpg?{time.time()}", diff_ann, title="Difference", column_index=2
    )

    DataJson().send_changes()
    report_per_image_images.loading = False


report_per_image_images = GridGallery(3, enable_zoom=True, sync_views=True, fill_rectangle=False)
report_per_image = Card(
    title="REPORT PER FRAME",
    description="Set filters to see frames with errors and click on the timeline to see the difference",
    content=Container(
        widgets=[
            Field(title="Select Video", content=report_per_image_video_select),
            Field(
                title="Timeline filters",
                description="Select metrics thresholds to display error frames on the timeline. Grey means there are no annotations on both frames for selected metrics",
                content=timeline_filters,
            ),
            Field(
                title="Timeline",
                content=Container(
                    widgets=[
                        timeline,
                        Text("<span>Frame:</span>"),
                        timeline_select_frame,
                    ],
                    direction="horizontal",
                    fractions=[1, 0, 0],
                    style="place-items: center;",
                ),
            ),
            report_per_image_table,
            report_per_image_images,
        ]
    ),
)

results = Container(
    widgets=[
        overall_stats,
        obj_count_per_class,
        geometry_quality,
        tags_stat,
        report_per_image,
    ],
    gap=10,
)


def get_overall_score(report, video_name=None):
    if video_name is None:
        d = report["overall"]
    else:
        d = report["per_video"][video_name]
    try:
        return d["overall-score"][0][("", "")]
    except KeyError:
        return 0


def get_obj_count_per_class_row(report, class_name, video_name=None):
    metrics = {
        "num-objects-gt": 0,
        "num-objects-pred": 0,
        "matches-recall": 1,
        "matches-precision": 1,
        "matches-f1": 1,
    }
    if video_name is None:
        d = report["overall"]
    else:
        d = report["per_video"][video_name]
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = d[metric_name][0][(class_name, "")]
        except KeyError:
            pass
    return [
        class_name,
        str(metrics["num-objects-gt"]),
        str(metrics["num-objects-pred"]),
        f'{int(metrics["matches-recall"]*metrics["num-objects-gt"])} of {metrics["num-objects-gt"]} ({round(metrics["matches-recall"]*100, 2)}%)',
        f'{int(metrics["matches-precision"]*metrics["num-objects-pred"])} of {metrics["num-objects-pred"]} ({round(metrics["matches-precision"]*100, 2)}%)',
        f'{round(metrics["matches-f1"]*100, 2)}%',
    ]


def get_average_f_measure_per_class(report, video_name=None):
    if video_name is None:
        d = report["overall"]
    else:
        d = report["per_video"][video_name]
    try:
        return d["matches-f1"][0][("", "")]
    except KeyError:
        return 1


def get_geometry_quality_row(report, class_name, video_name=None):
    metrics = {
        "pixel-accuracy": 1,
        "iou": 1,
    }
    if video_name is None:
        d = report["overall"]
    else:
        d = report["per_video"][video_name]
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = d[metric_name][0][(class_name, "")]
        except KeyError:
            pass

    return [
        class_name,
        f'{round(metrics["pixel-accuracy"]*100, 2)}%',
        f'{round(metrics["iou"]*100, 2)}%',
    ]


def get_average_iou(report, video_name=None):
    if video_name is None:
        d = report["overall"]
    else:
        d = report["per_video"][video_name]
    try:
        return d["iou"][0][("", "")]
    except KeyError:
        return 1


def get_tags_stat_table_row(report, tag_name, video_name=None):
    metrics = {
        "tags-total-gt": 0,
        "tags-total-pred": 0,
        "tags-precision": 1,
        "tags-recall": 1,
        "tags-f1": 1,
    }
    if video_name is None:
        d = report["overall"]
    else:
        d = report["per_video"][video_name]
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = d[metric_name][0][("", tag_name)]
        except KeyError:
            pass
    return [
        tag_name,
        metrics["tags-total-gt"],
        metrics["tags-total-pred"],
        f'{int(metrics["tags-precision"]*metrics["tags-total-pred"])} of {metrics["tags-total-pred"]} ({round(metrics["tags-precision"]*100, 2)}%)',
        f'{int(metrics["tags-recall"]*metrics["tags-total-gt"])} of {metrics["tags-total-gt"]} ({round(metrics["tags-recall"]*100, 2)}%)',
        f'{round(metrics["tags-f1"]*100, 2)}%',
    ]


def get_tags_values_stat_table_rows(report, tag_name, video_name=None):
    if video_name is None:
        d = report["overall"]
    else:
        d = report["per_video"][video_name]

    def get_one_row(tag_name, tag_value):
        metrics = {
            "tags-total-gt": 0,
            "tags-total-pred": 0,
            "tags-precision": 1,
            "tags-recall": 1,
            "tags-f1": 1,
        }
        for metric_name in metrics.keys():
            try:
                metrics[metric_name] = d[metric_name][0][
                    ("", json.dumps({"name": tag_name, "value": tag_value}))
                ]
            except KeyError:
                pass
        return [
            f"{tag_value}",
            metrics["tags-total-gt"],
            metrics["tags-total-pred"],
            f'{int(metrics["tags-precision"]*metrics["tags-total-pred"])} of {metrics["tags-total-pred"]} ({round(metrics["tags-precision"]*100, 2)}%)',
            f'{int(metrics["tags-recall"]*metrics["tags-total-gt"])} of {metrics["tags-total-gt"]} ({round(metrics["tags-recall"]*100, 2)}%)',
            f'{round(metrics["tags-f1"]*100, 2)}%',
        ]

    for t in d["tags-total-gt"][0]:
        try:
            t_json = json.loads(t[1])
            if t_json["name"] == tag_name:
                yield get_one_row(tag_name, t_json["value"])
        except:
            pass


def get_average_f_measure_per_tags(report, video_name=None):
    if video_name is None:
        d = report["overall"]
    else:
        d = report["per_video"][video_name]
    try:
        return d["tags-f1"][0][("", "")]
    except KeyError:
        return 1


def get_report_per_image_row_values(report, video_name, frame_n):
    metrics = {
        "matches-f1": 0.0,
        "matches-false-negative": 0,
        "matches-false-positive": 0,
        "tags-f1": 0.0,
        "tags-false-negative": 0,
        "tags-false-positive": 0,
        "iou": 0.0,
        "overall-score": 0.0,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report["per_video"][video_name][metric_name][frame_n][("", "")]
        except KeyError:
            pass
    return [
        metrics["matches-f1"] * 100,
        metrics["matches-false-negative"],
        metrics["matches-false-positive"],
        metrics["tags-f1"] * 100,
        metrics["tags-false-negative"],
        metrics["tags-false-positive"],
        metrics["iou"] * 100,
        metrics["overall-score"] * 100,
    ]


def get_report_per_image_row(report, video_name, frame_n):
    values = get_report_per_image_row_values(report, video_name, frame_n)
    return [
        frame_n,
        f"{round(values[0], 2)}%",
        values[1],
        values[2],
        f"{round(values[3], 2)}%",
        values[4],
        values[5],
        f"{round(values[6], 2)}%",
        f"{round(values[7], 2)}%",
    ]


def clean_up():
    report_per_image_images.clean_up()
    obj_count_per_class_table.read_json(
        {
            "columns": obj_count_per_class_table_columns,
            "data": [],
        }
    )
    obj_count_per_class_last.set(
        text=f"<b>Objects score (average F-measure) {0.00}%</b>", status="text"
    )

    geometry_quality_table.read_json(
        {
            "columns": geometry_quality_table_columns,
            "data": [],
        }
    )
    geometry_quality_last.set(f"<b>Geometry score (average IoU) {0.00}%</b>", status="text")

    tags_stat_table.read_json(
        {
            "columns": tags_stat_table_columns,
            "data": [],
        }
    )
    tags_stat_last.set("<b>Tags score (average F-measure) 0%</b>", status="text")

    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [["" for _ in report_per_image_table_columns]],
        }
    )

    status.set("-", status="text")


def unite_ranges(ranges: List[List[int]]):
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    res = [ranges[0]]
    for i in range(1, len(ranges)):
        if ranges[i][0] <= res[-1][1] + 1:
            res[-1][1] = max(res[-1][1], ranges[i][1])
        else:
            res.append(ranges[i])
    return res


def get_intervals_with_colors(report: dict, filters: dict = None, frame_range=[1, 1]):
    def get_color(frame_n):
        values = get_report_per_image_row_values(report, selected_video, frame_n)
        current_color = EMPTY_TIMELINE_COLOR
        if any(
            filters[filter_name] is not None
            for filter_name in ("objects_score", "objects_fn", "objects_fp")
        ):
            if any(values[0:3]):
                if filters["objects_score"] is not None and values[0] < filters["objects_score"]:
                    return ERROR_TIMELINE_COLOR
                if filters["objects_fn"] is not None and values[1] > filters["objects_fn"]:
                    return ERROR_TIMELINE_COLOR
                if filters["objects_fp"] is not None and values[2] > filters["objects_fp"]:
                    return ERROR_TIMELINE_COLOR
                current_color = GOOD_TIMELINE_COLOR
        if any(
            filters[filter_name] is not None for filter_name in ("tags_score", "tags_fn", "tags_fp")
        ):
            if any(values[3:6]):
                if filters["tags_score"] is not None and values[3] < filters["tags_score"]:
                    return ERROR_TIMELINE_COLOR
                if filters["tags_fn"] is not None and values[4] > filters["tags_fn"]:
                    return ERROR_TIMELINE_COLOR
                if filters["tags_fp"] is not None and values[5] > filters["tags_fp"]:
                    return ERROR_TIMELINE_COLOR
                current_color = GOOD_TIMELINE_COLOR
        if filters["geometry_score"] is not None:
            if any(v != 0 for v in values[0:3]):
                if values[6] < filters["geometry_score"]:
                    return ERROR_TIMELINE_COLOR
                current_color = GOOD_TIMELINE_COLOR
        if filters["overall_score"] is not None:
            if any(v != 0 for v in values[0:7]):
                if values[7] < filters["overall_score"]:
                    return ERROR_TIMELINE_COLOR
                current_color = GOOD_TIMELINE_COLOR

        return current_color

    colors = {
        ERROR_TIMELINE_COLOR: [],
        GOOD_TIMELINE_COLOR: [],
        EMPTY_TIMELINE_COLOR: [],
    }
    for frame_n in range(frame_range[0], frame_range[1] + 1):
        color = get_color(frame_n)
        colors[color].append([frame_n - frame_range[0], frame_n - frame_range[0]])

    for color, ranges in colors.items():
        united_ranges = unite_ranges(ranges)
        colors[color] = united_ranges
    return (
        [range for color, ranges in colors.items() for range in ranges],
        [color for color, ranges in colors.items() for range in ranges],
    )


def calculate_report(exam: Exam, attempt: Exam.ExamUser.Attempt, progress: Progress):
    return_button.disable()
    class_mapping = {
        obj_class.name: obj_class.name for obj_class in attempt.project_meta.obj_classes
    }
    gt_video_infos = get_vid_infos(exam.benchmark_dataset)
    progress.show()
    with progress(
        message="Calculating report...", total=sum(vid.frames_count for vid in gt_video_infos)
    ) as pbar:
        pred_video_infos = get_vid_infos(attempt.dataset)
        gt_video_anns = [
            sly.VideoAnnotation.from_json(
                ann_json, exam.benchmark_project_meta, key_id_map=sly.KeyIdMap()
            )
            for ann_json in g.api.video.annotation.download_bulk(
                exam.benchmark_dataset.id, [vid.id for vid in get_vid_infos(exam.benchmark_dataset)]
            )
        ]
        pred_video_anns = [
            sly.VideoAnnotation.from_json(ann_json, attempt.project_meta, key_id_map=sly.KeyIdMap())
            for ann_json in g.api.video.annotation.download_bulk(
                attempt.dataset.id, [vid.id for vid in get_vid_infos(attempt.dataset)]
            )
        ]
        report, diffs = calculate_exam_report(
            gt_video_infos=gt_video_infos,
            pred_video_infos=pred_video_infos,
            gt_video_anns=gt_video_anns,
            pred_video_anns=pred_video_anns,
            class_mapping=class_mapping,
            tags_whitelist=[tm.name for tm in attempt.project_meta.tag_metas],
            obj_tags_whitelist=[tm.name for tm in attempt.project_meta.tag_metas],
            iou_threshold=exam.iou_threshold() / 100,
            segmentation_mode=exam.segmentation_mode,
            progress=pbar,
        )

    # upload diff annotations
    with progress(message="Creating diff project...", total=1) as pbar:
        diff_project, diff_meta = get_or_create_diff_project(
            attempt.project, exam.attempt_project_meta
        )
        diff_dataset = get_or_create_diff_dataset(diff_project, attempt.dataset)
        pbar.update(1)
    with progress(message="Uploading diff annotations...", total=len(diffs)) as pbar:
        diff_video_infos = get_vid_infos(diff_dataset)
        diff_video_name_to_info = {video_info.name: video_info for video_info in diff_video_infos}
        error_obj_class = diff_meta.obj_classes.get("Error")
        if error_obj_class is None:
            error_obj_class = diff_meta.obj_classes.get("error")
        if error_obj_class is None:
            error_obj_class = sly.ObjClass("Error", sly.Bitmap, color=[255, 0, 0])
            diff_meta = diff_meta.add_obj_class(error_obj_class)
            g.api.project.update_meta(diff_project.id, diff_meta)
        for pred_video_info, vid_diffs in zip(pred_video_infos, diffs):
            diff_video_info = diff_video_name_to_info[pred_video_info.name]
            frame_size = (diff_video_info.frame_height, diff_video_info.frame_width)
            video_object = sly.VideoObject(error_obj_class)
            objects = sly.VideoObjectCollection([video_object])
            frames = sly.FrameCollection()
            for frame_index, diff in enumerate(vid_diffs):
                if diff is None:
                    continue
                figure = sly.VideoFigure(video_object, geometry=diff, frame_index=frame_index)
                frame = sly.Frame(frame_index, figures=[figure])
                frames.add(frame)

            ann = sly.VideoAnnotation(
                frame_size,
                frames_count=diff_video_info.frames_count,
                objects=objects,
                frames=frames,
            )
            g.api.video.annotation.append(diff_video_info.id, ann)
            pbar.update(1)

    return_button.enable()
    progress.hide()
    return report


def render_report(
    report,
    exam: Exam,
    user: Exam.ExamUser,
    attempt: Exam.ExamUser.Attempt,
    progress: Progress,
):
    results.loading = True
    return_button.disable()
    diff_project = get_diff_project(attempt.project)
    diff_dataset = get_diff_dataset(diff_project)

    if diff_project is None or diff_dataset is None:
        sly.logger.warning("Difference dataset not found. Recalculating report...")
        report = calculate_report(
            exam=exam,
            attempt=attempt,
            progress=progress,
        )
        diff_project = get_diff_project(attempt.project)
        diff_dataset = get_diff_dataset(diff_project)
        if diff_project is None or diff_dataset is None:
            raise RuntimeError("Difference dataset not found after recalculation")

    if error_in_report(report):
        error_text = report["error"]
        results.hide()
        results.loading = False
        return_button.enable()
        error_notification.set(title="Error in Report", description=error_text)
        error_notification.show()
        return

    report = report_to_dict(report)
    global current_report
    current_report = copy.deepcopy(report)

    passmark = exam.get_passmark()
    overall_score = get_overall_score(report)
    assigned_to.set(g.users.get(user.user_id).login, status="text")
    exam_passmark.set(f"{passmark}%", status="text")
    exam_score.set(f"{round(overall_score*100, 2)}%", status="text")
    status.set(
        (
            '<span style="color: green;">PASSED</span>'
            if overall_score * 100 > passmark
            else '<span style="color: red;">FAILED</span>'
        ),
        status="text",
    )
    benchmark_project_thumbnail.set(exam.benchmark_project)

    _render_report(
        report,
        exam,
        attempt,
        diff_project,
        diff_dataset,
    )

    results.loading = False
    return_button.enable()


@sly.timeit
def _render_report(
    report,
    exam: Exam,
    attempt: Exam.ExamUser.Attempt,
    diff_project: sly.ProjectInfo,
    diff_dataset: sly.DatasetInfo,
):
    # load videos
    global gt_vids
    global pred_vids
    global diff_vids
    gt_vids = get_vid_infos(exam.benchmark_dataset)
    pred_vids = get_vid_infos(attempt.dataset)
    diff_vids = get_vid_infos(diff_dataset)

    # load annotations
    global gt_annotations
    global pred_annotations
    global diff_annotations
    gt_annotations = [
        sly.VideoAnnotation.from_json(
            ann_json, exam.benchmark_project_meta, key_id_map=sly.KeyIdMap()
        )
        for ann_json in g.api.video.annotation.download_bulk(
            exam.benchmark_dataset.id, [vid.id for vid in gt_vids]
        )
    ]
    pred_annotations = [
        sly.VideoAnnotation.from_json(ann_json, attempt.project_meta, key_id_map=sly.KeyIdMap())
        for ann_json in g.api.video.annotation.download_bulk(
            attempt.dataset.id, [vid.id for vid in pred_vids]
        )
    ]
    diff_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(diff_project.id))
    diff_annotations = [
        sly.VideoAnnotation.from_json(ann_json, diff_meta, key_id_map=sly.KeyIdMap())
        for ann_json in g.api.video.annotation.download_bulk(
            diff_dataset.id, [vid.id for vid in diff_vids]
        )
    ]

    classes = [obj_class.name for obj_class in attempt.project_meta.obj_classes]
    # obj count per class
    obj_count_per_class_table.read_json(
        {
            "columns": obj_count_per_class_table_columns,
            "data": [get_obj_count_per_class_row(report, cls_name) for cls_name in classes],
        }
    )
    obj_count_per_class_last.set(
        text=f"<b>Objects score (average F-measure) {round(get_average_f_measure_per_class(report)*100, 2)}%</b>",
        status="text",
    )

    # geometry quality
    geometry_quality_table.read_json(
        {
            "columns": geometry_quality_table_columns,
            "data": [get_geometry_quality_row(report, cls_name) for cls_name in classes],
        }
    )
    geometry_quality_last.set(
        f"<b>Geometry score (average IoU) {round(get_average_iou(report)*100, 2)}%</b>",
        status="text",
    )

    # tags
    tags = [tm.name for tm in attempt.project_meta.tag_metas]
    tags_stat_table.read_json(
        {
            "columns": tags_stat_table_columns,
            "data": [get_tags_stat_table_row(report, tag_name) for tag_name in tags],
        }
    )
    tags_stat_last.set(
        f"<b>Tags score (average F-measure) {round(get_average_f_measure_per_tags(report)*100, 2)}%</b>",
        status="text",
    )

    if len(tags) > 0:
        tag_name = tags[0]
        tags_values_stat_table.read_json(
            {
                "columns": tags_values_stat_table_columns,
                "data": [row for row in get_tags_values_stat_table_rows(report, tag_name)],
            }
        )
    else:
        tags_values_stat_table.read_json(
            {
                "columns": tags_values_stat_table_columns,
                "data": [],
            }
        )

    # set timeline
    report_per_image_video_select.set(
        items=[Select.Item(video.name, video.name) for video in pred_vids]
    )
    apply_timeline_filters()


@timeline.click
def timeline_click_cb(pointer):
    frame_n = pointer + current_frame_range[0]
    timeline_select_frame.value = frame_n
    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [get_report_per_image_row(current_report, selected_video, frame_n)],
        }
    )
    show_images(frame_n)


def apply_timeline_filters(*args):
    filters = get_timeline_filters()
    intervals, colors = get_intervals_with_colors(current_report, filters, current_frame_range)
    timeline.set(current_frame_range[1] - current_frame_range[0] + 1, intervals, colors)


for filter_metric_widget in timeline_filters_widgets.values():

    def cb_factory(filter_metric_widget):
        def metric_switched_cb(is_switched):
            if is_switched:
                filter_metric_widget["input"].enable()
            else:
                filter_metric_widget["input"].disable()
            apply_timeline_filters()

        return metric_switched_cb

    filter_metric_widget["input"].value_changed(apply_timeline_filters)
    filter_metric_widget["switch"].value_changed(cb_factory(filter_metric_widget))
    filter_metric_widget["input"].disable()
