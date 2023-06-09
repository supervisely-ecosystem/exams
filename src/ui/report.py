import json
import time
import supervisely as sly
from supervisely.app.widgets import (
    Container,
    Text,
    Card,
    Button,
    Field,
    ProjectThumbnail,
    Flexbox,
    Table,
    GridGallery,
)
import src.globals as g
from src.exam import calculate_exam_report

from supervisely.app import DataJson


gt_imgs = []
pred_imgs = []
diff_imgs = []
gt_img_anns = {}
pred_img_anns = {}
diff_img_anns = {}

assigned_to = Text("")
exam_passmark = Text("")
exam_score = Text("")
status = Text("")
benchmark_project_thumbnail = ProjectThumbnail()

overall_stats = Card(
    title="EXAM OVERALL STATS",
    content=Flexbox(
        widgets=[
            Field(title="Assigned to", content=assigned_to),
            Field(title="Exam Passmark", content=exam_passmark),
            Field(title="Exam Score", content=exam_score),
            Field(title="Status", content=status),
            Field(title="Benchmark Project", content=benchmark_project_thumbnail),
        ],
        gap=45,
    ),
)

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
    content=Container(
        widgets=[obj_count_per_class_table, obj_count_per_class_last], gap=5
    ),
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
tags_stat = Card(
    title="TAGS", content=Container(widgets=[tags_stat_table, tags_stat_last], gap=5)
)

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

return_button = Button(
    "Return to Exams", button_size="small", icon="zmdi zmdi-arrow-left"
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
layout = Container(widgets=[return_button, results], gap=20)


def get_overall_score(result):
    for data in result:
        if data["metric_name"] == "overall-score":
            if (
                data["class_gt"] == ""
                and data["image_gt_id"] == 0
                and data["tag_name"] == ""
            ):
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
            if (
                data["metric_name"] == "num-objects-gt"
                and data["class_gt"] == class_name
            ):
                num_objects_gt = data["value"]
            if (
                data["metric_name"] == "num-objects-pred"
                and data["class_gt"] == class_name
            ):
                num_objects_pred = data["value"]
            if (
                data["metric_name"] == "matches-recall"
                and data["class_gt"] == class_name
            ):
                matches_recall_percent = data["value"]
            if (
                data["metric_name"] == "matches-precision"
                and data["class_gt"] == class_name
            ):
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
            if (
                data["metric_name"] == "pixel-accuracy"
                and data["class_gt"] == class_name
            ):
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
            if (
                data["metric_name"] == "tags-total-pred"
                and data["tag_name"] == tag_name
            ):
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
        if (
            data["image_gt_id"] == image_id
            and data["class_gt"] == ""
            and data["tag_name"] == ""
        ):
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
    geometry_quality_last.set(
        f"<b>Geometry score (average IoU) {0.00}%</b>", status="text"
    )

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


def get_diff_project(exam_project: sly.ProjectInfo):
    diff_project = g.api.project.get_info_by_name(
        exam_project.workspace_id, f"{exam_project.name}_DIFF"
    )
    if diff_project is None:
        return None, None
    diff_dataset = g.api.dataset.get_list(diff_project.id)[0]
    return diff_project, diff_dataset


def create_diff_project(exam_project: sly.ProjectInfo, exam_dataset: sly.DatasetInfo):
    # create diff project
    project_diff_info = g.api.project.create(
        workspace_id=exam_project.workspace_id,
        name=exam_project.name + "_DIFF",
        type=exam_project.type,
        description="",
        change_name_if_conflict=True,
    )

    # upload custom_data
    g.api.project.update_custom_data(
        project_diff_info.id, {"exam_project_id": exam_project.id}
    )

    # upload diff project meta
    g.api.project.merge_metas(exam_project.id, project_diff_info.id)

    # create diff dataset
    dataset_diff_info = g.api.dataset.create(project_diff_info.id, exam_dataset.name)

    # upload diff images
    imgs = g.api.image.get_list(exam_dataset.id)
    img_ids, img_names = zip(*[(img.id, img.name) for img in imgs])
    g.api.image.upload_ids(dataset_diff_info.id, img_names, img_ids)

    return project_diff_info, dataset_diff_info


def get_or_create_diff_project(
    exam_project: sly.ProjectInfo, exam_dataset: sly.DatasetInfo
):
    diff_project, diff_dataset = get_diff_project(exam_project)
    if diff_project is None or diff_dataset is None:
        return create_diff_project(exam_project, exam_dataset)
    return diff_project, diff_dataset


@sly.timeit
def calculate_report(
    benchmark_dataset,
    exam_project,
    exam_dataset,
    classes_whitelist,
    tags_whitelist,
    obj_tags_whitelist,
    iou_threshold,
):
    return_button.disable()
    class_matches = [{"class_gt": v, "class_pred": v} for v in classes_whitelist]
    diff_project, diff_dataset = get_or_create_diff_project(exam_project, exam_dataset)

    report = calculate_exam_report(
        server_address=g.server_address,
        api_token=g.api_token,
        project_gt_id=benchmark_dataset.project_id,
        dataset_gt_id=benchmark_dataset.id,
        project_pred_id=exam_dataset.project_id,
        dataset_pred_id=exam_dataset.id,
        project_dest_id=diff_project.id,
        dataset_dest_id=diff_dataset.id,
        class_matches=class_matches,
        tags_whitelist=tags_whitelist,
        obj_tags_whitelist=obj_tags_whitelist,
        iou_threshold=iou_threshold / 100,
    )
    return_button.enable()

    return report


def save_report(report, attempt):
    with open("report.json", "w") as f:
        json.dump(report, f)
    report_path = f"/exam_data/{attempt.project.id}/report.json"
    g.api.file.upload(g.team_id, "report.json", report_path)


def get_report(workspace_id, project_id):
    while g.is_refreshing_report:
        time.sleep(0.2)
    report_path = f"/exam_data/{workspace_id}/{project_id}/report.json"
    if g.api.file.exists(g.team_id, report_path):
        g.api.file.download(g.team_id, report_path, "report.json")
        with open("report.json", "r") as f:
            report = json.load(f)
        return report
    return None


def update_report_status(report, attempt):
    def get_overall_score(report):
        for data in report:
            if data["metric_name"] == "overall-score":
                if (
                    data["class_gt"] == ""
                    and data["image_gt_id"] == 0
                    and data["tag_name"] == ""
                ):
                    return data["value"]
        return 0

    custom_data = g.api.project.get_info_by_id(attempt.project.id).custom_data
    custom_data["overall_score"] = get_overall_score(report)
    g.api.project.update_custom_data(attempt.project.id, custom_data)


def refresh_report(value_dict):
    g.is_refreshing_report = True

    workspace_id = value_dict["workspace_id"]
    user_id = value_dict["user_id"]

    benchmark_dataset = g.exams[workspace_id].benchmark_dataset
    attempt = g.exams[workspace_id].users[user_id].attempts[0]
    iou_threshold = g.exams[workspace_id].benchmark_project.custom_data["threshold"]
    attempt_project_meta = attempt.project_meta

    report = calculate_report(
        benchmark_dataset,
        attempt.project,
        attempt.dataset,
        classes_whitelist=[oc.name for oc in attempt_project_meta.obj_classes],
        tags_whitelist=[tm.name for tm in attempt_project_meta.tag_metas],
        obj_tags_whitelist=[tm.name for tm in attempt_project_meta.tag_metas],
        iou_threshold=iou_threshold,
    )

    save_report(report, attempt)
    update_report_status(report, attempt)
    g.is_refreshing_report = False

    return report


def render_report(
    report,
    benchmark_project,
    benchmark_dataset,
    exam_project,
    exam_dataset,
    classes,
    tags,
    passmark,
    iou_threshold,
    user_name,
):
    results.loading = True
    return_button.disable()

    diff_project, diff_dataset = get_diff_project(exam_project)
    if diff_project is None or diff_dataset is None:
        sly.logger.warning("Difference project not found. Recalculating report...")
        report = calculate_report(
            benchmark_dataset=benchmark_dataset,
            exam_dataset=exam_dataset,
            classes_whitelist=classes,
            tags_whitelist=tags,
            obj_tags_whitelist=tags,
            iou_threshold=iou_threshold,
        )
        diff_project, diff_dataset = get_diff_project(exam_project)
        if diff_project is None or diff_dataset is None:
            raise RuntimeError("Difference project not found after recalculation")

    overall_score = get_overall_score(report)
    assigned_to.set(user_name, status="text")
    exam_passmark.set(f"{passmark}%", status="text")
    exam_score.set(f"{round(overall_score*100, 2)}%", status="text")
    status.set(
        '<span style="color: green;">PASSED</span>'
        if overall_score * 100 > passmark
        else '<span style="color: red;">FAILED</span>',
        status="text",
    )
    benchmark_project_thumbnail.set(benchmark_project)

    # load images
    global gt_imgs
    gt_imgs = g.api.image.get_list(benchmark_dataset.id)
    global pred_imgs
    pred_imgs = g.api.image.get_list(exam_dataset.id)
    global diff_imgs
    diff_imgs = g.api.image.get_list(diff_dataset.id)

    # load image annotations
    project_gt_meta = sly.ProjectMeta.from_json(
        g.api.project.get_meta(benchmark_project.id)
    )
    project_pred_meta = sly.ProjectMeta.from_json(
        g.api.project.get_meta(exam_project.id)
    )
    diff_project_meta = sly.ProjectMeta.from_json(
        g.api.project.get_meta(diff_project.id)
    )
    global gt_img_anns
    gt_img_anns = {
        ann_info.image_id: sly.Annotation.from_json(
            ann_info.annotation, project_gt_meta
        )
        for ann_info in g.api.annotation.get_list(benchmark_dataset.id)
    }
    global pred_img_anns
    pred_img_anns = {
        ann_info.image_id: sly.Annotation.from_json(
            ann_info.annotation, project_pred_meta
        )
        for ann_info in g.api.annotation.get_list(exam_dataset.id)
    }
    global diff_img_anns
    diff_img_anns = {
        ann_info.image_id: sly.Annotation.from_json(
            ann_info.annotation, diff_project_meta
        )
        for ann_info in g.api.annotation.get_list(diff_dataset.id)
    }

    # obj count per class
    obj_count_per_class_table.read_json(
        {
            "columns": obj_count_per_class_table_columns,
            "data": [
                get_obj_count_per_class_row(report, cls_name) for cls_name in classes
            ],
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
            "data": [
                get_geometry_quality_row(report, cls_name) for cls_name in classes
            ],
        }
    )
    geometry_quality_last.set(
        f"<b>Geometry score (average IoU) {round(get_average_iou(report)*100, 2)}%</b>",
        status="text",
    )

    # tags
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
                get_report_per_image_row(report, gt_img.name, gt_img.id)
                for gt_img in gt_imgs
            ],
        }
    )

    results.loading = False
    return_button.enable()
