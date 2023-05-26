import json
import time
from typing import List
import supervisely as sly

import src.globals as g


def create_exam_workspace(exam_name: str):
    ws = g.api.workspace.create(
        team_id=g.team_id,
        name=f'Exam: "{exam_name}"',
        description="",
        change_name_if_conflict=False,
    )
    return ws


def get_or_create_exam_workspace(exam_name: str):
    for ws in g.api.workspace.get_list(g.team_id):
        if ws.name == f'Exam: "{exam_name}"':
            return ws
    return create_exam_workspace(exam_name)


def create_exam_project_for_user(
    workspace: sly.api.workspace_api.WorkspaceInfo, user: sly.api.user_api.UserInfo
):
    return g.api.project.create(
        workspace_id=workspace.id,
        name=f"{workspace.name}. User: {user.name} ({user.email})",
        description="",
        change_name_if_conflict=False,
    )


def get_or_create_exam_project_for_user(
    workspace: sly.api.workspace_api.WorkspaceInfo, user: sly.api.user_api.UserInfo
):
    for pr in g.api.project.get_list(workspace.id):
        if pr.name == f"{workspace.name}. User: {user.name} ({user.email})":
            return pr
    return create_exam_project_for_user(workspace, user)


def create_project_meta(
    benchmark_project_meta: sly.ProjectMeta,
    classes_whitelist: List[str],
    tags_whitelist: List[str],
):
    return benchmark_project_meta.clone(
        obj_classes=[
            obj_class
            for obj_class in benchmark_project_meta.obj_classes
            if obj_class.name in classes_whitelist
        ],
        tag_metas=[
            tag_meta
            for tag_meta in benchmark_project_meta.tag_metas
            if tag_meta.name in tags_whitelist
        ],
    )


def create_labeling_job(
    project: sly.ProjectInfo,
    dataset: sly.DatasetInfo,
    user_ids,
    readme,
    description,
    classes_to_label,
    tags_to_label,
    reviewer_id,
):
    lj = g.api.labeling_job.create(
        name=project.name,
        dataset_id=dataset.id,
        user_ids=user_ids,
        readme=readme,
        description=description,
        classes_to_label=classes_to_label,
        tags_to_label=tags_to_label,
        reviewer_id=reviewer_id,
        images_ids=[img.id for img in g.api.image.get_list(dataset.id)],
    )
    return lj


def get_exams_workspaces():
    return [ws for ws in g.api.workspace.get_list(g.team_id) if ws.name.startswith('Exam: "')]


def get_exam_benchmark_project(workspace: sly.api.workspace_api.WorkspaceInfo):
    return g.exams[workspace.id]["benchmark_project"]


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def create_diff_project(exam_project):
    # create diff project
    project_diff_info = g.api.project.create(
        workspace_id=exam_project.workspace_id,
        name=exam_project.name + "_DIFF",
        type=exam_project.type,
        description="",
        change_name_if_conflict=True,
    )
    # upload diff project meta
    g.api.project.merge_metas(exam_project.id, project_diff_info.id)
    
    # create diff dataset
    exam_dataset = get_dataset_by_project_id(exam_project.id)
    dataset_diff_info = g.api.dataset.create(
        project_diff_info.id, exam_dataset.name
    )

    # upload diff images
    imgs = g.api.image.get_list(exam_dataset.id)
    img_ids, img_names = zip(*[(img.id, img.name) for img in imgs])
    diff_imgs = g.api.image.upload_ids(dataset_diff_info.id, img_names, img_ids)

    return project_diff_info, dataset_diff_info


def get_diff_project(exam_project: sly.ProjectInfo):
    diff_project = g.api.project.get_info_by_name(exam_project.workspace_id, f"{exam_project.name}_DIFF")
    if diff_project is None:
        return None, None
    diff_dataset = g.api.dataset.get_list(diff_project.id)[0]
    return diff_project, diff_dataset


def get_or_create_diff_project(exam_project: sly.ProjectInfo):
    diff_project, diff_dataset = get_diff_project(exam_project)
    if diff_project is None or diff_dataset is None:
        return create_diff_project(exam_project)
    return diff_project, diff_dataset


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


def save_report(report, workspace_id, project_id):
    with open("report.json", "w") as f:
        json.dump(report, f, indent=4)
    report_path = f"/exam_data/{workspace_id}/{project_id}/report.json"
    g.api.file.upload(g.team_id, "report.json", report_path)


def get_workspace_by_project_id(project_id):
    for exam in g.exams.values():
        if exam["benchmark_project"].id == project_id:
            return exam["workspace_info"]
        for project in exam["exam_projects"]:
            if project.id == project_id:
                return exam["workspace_info"]


def get_labeling_jobs(project_id):
    workspace_id = get_workspace_by_project_id(project_id).id
    return g.exams[workspace_id]["labeling_jobs"][project_id]


def update_report_status(report, project_id):
    def get_overall_score(report):
        for data in report:
            if data["metric_name"] == "overall-score":
                if data["class_gt"] == "" and data["image_gt_id"] == 0 and data["tag_name"] == "":
                    return data["value"]
        return 0
    
    labeling_jobs = sorted(get_labeling_jobs(project_id), key=lambda lj: lj.id, reverse=True)
    custom_data = g.api.project.get_info_by_id(project_id).custom_data
    custom_data["overall_score"] = get_overall_score(report)
    custom_data["last_labeling_job_id"] = labeling_jobs[0].id
    g.api.project.update_custom_data(project_id, custom_data)


def utc_to_human(utc):
    if utc is None:
        return None
    import dateutil.parser as p
    d = p.parse(utc)
    return d.strftime("%d %b %y")


def get_image_tool_url(server_address, team_id, workspace_id, project_id, dataset_id, image_id):
    return f"{server_address}/app/images/{team_id}/{workspace_id}/{project_id}/{dataset_id}/#image-{image_id}"


def get_project_by_dataset_id(dataset_id):
    for exam in g.exams.values():
        if exam["benchmark_dataset"].id == dataset_id:
            return exam["benchmark_project"]
        for dataset in exam["exam_datasets"]:
            if dataset.id == dataset_id:
                for project in exam["exam_projects"]:
                    if project.id == dataset.project_id:
                        return project
    return None


def get_dataset_by_id(dataset_id):
    for exam in g.exams.values():
        if exam["benchmark_dataset"].id == dataset_id:
            return exam["benchmark_dataset"]
        for dataset in exam["exam_datasets"]:
            if dataset.id == dataset_id:
                return dataset
    return None


def get_dataset_by_project_id(project_id):
    for exam in g.exams.values():
        if exam["benchmark_project"].id == project_id:
            return exam["benchmark_dataset"]
        for dataset in exam["exam_datasets"]:
            if dataset.project_id == project_id:
                return dataset
    return None