import json
import os
import time

from supervisely.app.widgets import (
    Button,
    NotificationBox,
    Text,
    ProjectThumbnail,
    Card,
    Field,
    Flexbox,
)
import supervisely as sly

import src.globals as g

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

return_button = Button("Return to Exams", button_size="small", icon="zmdi zmdi-arrow-left")
error_notification = NotificationBox(title="Report Error", box_type="error", description="")
error_notification.hide()


def save_report(report, attempt, progress=None):
    with open("report.json", "w") as f:
        json.dump(report, f)
    report_path = f"/exam_data/{attempt.project.id}/report.json"
    pbar = None
    if progress is not None:
        progress.show()
        pbar = progress(message="Uploading report...", total=sly.fs.get_file_size("report.json"))
    g.api.file.upload(g.team_id, "report.json", report_path, progress_cb=pbar)
    if progress is not None:
        progress.hide()


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


def error_in_report(report):
    if isinstance(report, dict):
        if "error" in report:
            return True
    return False


def get_diff_project(attempt_project: sly.ProjectInfo):
    diff_project = g.api.project.get_info_by_name(
        attempt_project.workspace_id, f"{attempt_project.name}_DIFF"
    )
    return diff_project


def get_diff_dataset(diff_project: sly.ProjectInfo):
    try:
        return g.api.dataset.get_list(diff_project.id)[0]
    except IndexError:
        return None


def create_diff_project(attempt_project: sly.ProjectInfo, attempt_meta: sly.ProjectMeta):
    # create diff project
    diff_project = g.api.project.create(
        workspace_id=attempt_project.workspace_id,
        name=attempt_project.name + "_DIFF",
        type=attempt_project.type,
        description="",
        change_name_if_conflict=True,
    )

    # upload custom_data
    g.api.project.update_custom_data(diff_project.id, {"attempt_project_id": attempt_project.id})

    # upload diff project meta
    diff_obj_class = sly.ObjClass("Error", sly.Bitmap, color=[255, 0, 0])
    diff_meta = attempt_meta.add_obj_class(diff_obj_class)
    g.api.project.update_meta(diff_project.id, diff_meta)

    return diff_project, diff_meta


def get_or_create_diff_project(attempt_project: sly.ProjectInfo, attempt_meta: sly.ProjectMeta):
    diff_project = get_diff_project(attempt_project)
    if diff_project is None:
        return create_diff_project(attempt_project, attempt_meta)
    diff_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(diff_project.id))
    return diff_project, diff_meta
