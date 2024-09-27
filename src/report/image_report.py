import json
import time
import supervisely as sly
from supervisely.app import DataJson
from supervisely.app.widgets import (
    Container,
    Text,
    Card,
    Table,
    GridGallery,
)

import src.globals as g
from src.report.image_metrics import calculate_exam_report
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
from src.exam import Exam


gt_imgs = []
pred_imgs = []
diff_imgs = []
gt_img_anns = {}
pred_img_anns = {}
diff_img_anns = {}

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
tags_stat = Card(title="TAGS", content=Container(widgets=[tags_stat_table, tags_stat_last], gap=5))

report_per_image_table_columns = [
    "NAME",
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


@report_per_image_table.click
def show_images(datapoint):
    global report_per_image_images
    report_per_image_images.clean_up()
    row = datapoint.row
    img_name = row["NAME"]

    def find_image_by_name(img_name, images) -> sly.ImageInfo:
        for image in images:
            if image.name == img_name:
                return image
        return None

    gt_img = find_image_by_name(img_name, gt_imgs)
    gt_ann = gt_img_anns[gt_img.id]
    pred_img = find_image_by_name(img_name, pred_imgs)
    pred_ann = pred_img_anns[pred_img.id]
    diff_img = find_image_by_name(img_name, diff_imgs)
    diff_ann = diff_img_anns[diff_img.id]

    report_per_image_images.append(
        gt_img.preview_url, gt_ann, title="Benchmark (Ground Truth)", column_index=0
    )
    report_per_image_images.append(
        pred_img.preview_url, pred_ann, title="Labeler output", column_index=1
    )
    report_per_image_images.append(
        diff_img.preview_url, diff_ann, title="Difference", column_index=2
    )

    DataJson().send_changes()


report_per_image_images = GridGallery(3)
report_per_image = Card(
    title="REPORT PER IMAGE",
    content=Container(
        widgets=[
            report_per_image_table,
            Card(content=report_per_image_images, collapsable=True),
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
    gap=20,
)


def get_overall_score(result):
    for data in result:
        if data["metric_name"] == "overall-score":
            if data["class_gt"] == "" and data["image_gt_id"] == 0 and data["tag_name"] == "":
                return data["value"]
    return 0


def get_obj_count_per_class_row(result, class_name):
    num_objects_gt = 0
    num_objects_pred = 0
    matches_recall_percent = 1
    matches_precision_percent = 1
    matches_f_measure = 1
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "num-objects-gt" and data["class_gt"] == class_name:
                num_objects_gt = data["value"]
            if data["metric_name"] == "num-objects-pred" and data["class_gt"] == class_name:
                num_objects_pred = data["value"]
            if data["metric_name"] == "matches-recall" and data["class_gt"] == class_name:
                matches_recall_percent = data["value"]
            if data["metric_name"] == "matches-precision" and data["class_gt"] == class_name:
                matches_precision_percent = data["value"]
            if data["metric_name"] == "matches-f1" and data["class_gt"] == class_name:
                matches_f_measure = data["value"]
    return [
        class_name,
        str(num_objects_gt),
        str(num_objects_pred),
        f"{int(matches_recall_percent*num_objects_gt)} of {num_objects_gt} ({round(matches_recall_percent*100, 2)}%)",
        f"{int(matches_precision_percent*num_objects_pred)} of {num_objects_pred} ({round(matches_precision_percent*100, 2)}%)",
        f"{round(matches_f_measure*100, 2)}%",
    ]


def get_average_f_measure_per_class(result):
    avg_f1 = 1
    f1_measures = []
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "matches-f1" and data["class_gt"] != "":
                f1_measures.append(data["value"])
    if len(f1_measures) > 0:
        avg_f1 = sum(f1_measures) / len(f1_measures)
    return avg_f1


def get_geometry_quality_row(result, class_name):
    pixel_accuracy = 1
    iou = 1
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "pixel-accuracy" and data["class_gt"] == class_name:
                pixel_accuracy = data["value"]
            if data["metric_name"] == "iou" and data["class_gt"] == class_name:
                iou = data["value"]

    return [class_name, f"{round(pixel_accuracy*100, 2)}%", f"{round(iou*100, 2)}%"]


def get_average_iou(result):
    avg_iou = 1
    iou = []
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "iou" and data["class_gt"] != "":
                iou.append(data["value"])
    if len(iou) > 0:
        avg_iou = sum(iou) / len(iou)
    return avg_iou


def get_tags_stat_table_row(result, tag_name):
    total_gt = 0
    total_pred = 0
    precision = 1
    recall = 1
    f1 = 1
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "tags-total-gt" and data["tag_name"] == tag_name:
                total_gt = data["value"]
            if data["metric_name"] == "tags-total-pred" and data["tag_name"] == tag_name:
                total_pred = data["value"]
            if data["metric_name"] == "tags-precision" and data["tag_name"] == tag_name:
                precision = data["value"]
            if data["metric_name"] == "tags-recall" and data["tag_name"] == tag_name:
                recall = data["value"]
            if data["metric_name"] == "tags-f1" and data["tag_name"] == tag_name:
                f1 = data["value"]

    return [
        tag_name,
        total_gt,
        total_pred,
        f"{int(precision*total_pred)} of {total_pred} ({round(precision*100, 2)}%)",
        f"{int(recall*total_gt)} of {total_gt} ({round(recall*100, 2)}%)",
        f"{round(f1*100, 2)}%",
    ]


def get_average_f_measure_per_tags(result):
    avg_f1 = 1
    f1_measures = []
    for data in result:
        if data["image_gt_id"] == 0:
            if data["metric_name"] == "tags-f1" and data["tag_name"] == "":
                f1_measures.append(data["value"])
    if len(f1_measures) > 0:
        avg_f1 = sum(f1_measures) / len(f1_measures)
    return avg_f1


def get_report_per_image_row(result, image_name, image_id):
    objects_score = 1
    objects_missing = None
    obj_false_positive = None
    tag_score = 1
    tag_missing = None
    tag_false_positive = None
    geometry_score = 0
    overall_score = 0
    for data in result:
        if data["image_gt_id"] == image_id and data["class_gt"] == "" and data["tag_name"] == "":
            if data["metric_name"] == "matches-f1":
                objects_score = data["value"]
            if data["metric_name"] == "matches-false-negative":
                objects_missing = data["value"]
            if data["metric_name"] == "matches-false-positive":
                obj_false_positive = data["value"]
            if data["metric_name"] == "tags-f1":
                tag_score = data["value"]
            if data["metric_name"] == "tags-false-negative":
                tag_missing = data["value"]
            if data["metric_name"] == "tags-false-positive":
                tag_false_positive = data["value"]
            if data["metric_name"] == "iou":
                geometry_score = data["value"]
            if data["metric_name"] == "overall-score":
                overall_score = data["value"]

    return [
        image_name,
        f"{round(objects_score*100, 2)}%",
        objects_missing,
        obj_false_positive,
        f"{round(tag_score*100, 2)}%",
        tag_missing,
        tag_false_positive,
        f"{round(geometry_score*100, 2)}%",
        f"{round(overall_score*100, 2)}%",
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
            "data": [],
        }
    )

    assigned_to.set("-", status="text")
    exam_passmark.set("-", status="text")
    exam_score.set("-", status="text")
    status.set("-", status="text")
    error_notification.hide()
    results.show()


def create_diff_dataset(diff_project: sly.ProjectInfo, attempt_dataset: sly.DatasetInfo):
    diff_dataset = g.api.dataset.create(diff_project.id, attempt_dataset.name)
    imgs = g.api.image.get_list(attempt_dataset.id)
    img_ids, img_names = zip(*[(img.id, img.name) for img in imgs])
    g.api.image.upload_ids(diff_dataset.id, img_names, img_ids)

    return diff_dataset


def get_or_create_diff_dataset(diff_project: sly.ProjectInfo, attempt_dataset: sly.DatasetInfo):
    diff_dataset = get_diff_dataset(diff_project)
    if diff_dataset is None:
        diff_dataset = create_diff_dataset(diff_project, attempt_dataset)
    return diff_dataset


def get_img_infos(dataset: sly.DatasetInfo):
    return sorted(g.api.image.get_list(dataset.id), key=lambda x: x.id)


def get_ann_infos(dataset: sly.DatasetInfo):
    return sorted(g.api.annotation.get_list(dataset.id), key=lambda x: x.image_id)


@sly.timeit
def calculate_report(exam: Exam, attempt: Exam.ExamUser.Attempt):
    return_button.disable()
    class_mapping = {
        obj_class.name: obj_class.name for obj_class in attempt.project_meta.obj_classes
    }
    report, diffs = calculate_exam_report(
        united_meta=attempt.project_meta,
        img_infos_gt=get_img_infos(exam.benchmark_dataset),
        img_infos_pred=get_img_infos(attempt.dataset),
        ann_infos_gt=get_ann_infos(exam.benchmark_dataset),
        ann_infos_pred=get_ann_infos(attempt.dataset),
        class_mapping=class_mapping,
        tags_whitelist=[tm.name for tm in attempt.project_meta.tag_metas],
        obj_tags_whitelist=[tm.name for tm in attempt.project_meta.tag_metas],
        iou_threshold=exam.iou_threshold() / 100,
        segmentation_mode=exam.segmentation_mode,
    )

    # upload diff annotations
    diff_project, diff_meta = get_or_create_diff_project(attempt.project, exam.attempt_project_meta)
    diff_dataset = get_or_create_diff_dataset(diff_project, attempt.dataset)
    error_obj_class = diff_meta.obj_classes.get("Error")
    if error_obj_class is None:
        error_obj_class = diff_meta.obj_classes.get("error")
    if error_obj_class is None:
        error_obj_class = sly.ObjClass("Error", sly.Bitmap, color=[255, 0, 0])
        diff_meta = diff_meta.add_obj_class(error_obj_class)
        g.api.project.update_meta(diff_project.id, diff_meta)
    for batch in sly.batched(list(zip(diffs, get_img_infos(diff_dataset)))):
        anns = []
        img_ids = []
        for diff, diff_img in batch:
            if diff is None:
                continue
            anns.append(
                sly.Annotation(
                    img_size=(diff_img.height, diff_img.width),
                    labels=[
                        sly.Label(
                            geometry=diff,
                            obj_class=error_obj_class,
                        )
                    ],
                )
            )
            img_ids.append(diff_img.id)
        g.api.annotation.upload_anns(img_ids, anns)

    return_button.enable()

    return report


def render_report(
    report,
    exam: Exam,
    user: Exam.ExamUser,
    attempt: Exam.ExamUser.Attempt,
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

    # load images
    global gt_imgs
    gt_imgs = get_img_infos(exam.benchmark_dataset)
    global pred_imgs
    pred_imgs = get_img_infos(attempt.dataset)
    global diff_imgs
    diff_imgs = get_img_infos(diff_dataset)

    # load image annotations
    diff_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(diff_project.id))
    global gt_img_anns
    gt_img_anns = {
        ann_info.image_id: sly.Annotation.from_json(ann_info.annotation, exam.attempt_project_meta)
        for ann_info in g.api.annotation.get_list(exam.benchmark_dataset.id)
    }
    global pred_img_anns
    pred_img_anns = {
        ann_info.image_id: sly.Annotation.from_json(ann_info.annotation, attempt.project_meta)
        for ann_info in g.api.annotation.get_list(attempt.dataset.id)
    }
    global diff_img_anns
    diff_img_anns = {
        ann_info.image_id: sly.Annotation.from_json(ann_info.annotation, diff_meta)
        for ann_info in g.api.annotation.get_list(diff_dataset.id)
    }

    # obj count per class
    classes = [obj_class.name for obj_class in attempt.project_meta.obj_classes]
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

    # per image
    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [
                get_report_per_image_row(report, gt_img.name, gt_img.id) for gt_img in gt_imgs
            ],
        }
    )

    results.loading = False
    return_button.enable()
