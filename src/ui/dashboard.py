import json
import supervisely as sly
from supervisely.app.widgets import Text, Container, Button, Empty

import src.globals as g
import src.utils as utils

from src.ui.expandable_table import ExpandableTable


exams_table = ExpandableTable()


def generate_exam_users_table_row_data(project: sly.ProjectInfo, workspace_id, passmark):
    status_text = {
        str(sly.api.labeling_job_api.LabelingJobApi.Status.PENDING): 'PENDING',
        str(sly.api.labeling_job_api.LabelingJobApi.Status.IN_PROGRESS): 'IN PROGRESS',
        str(sly.api.labeling_job_api.LabelingJobApi.Status.ON_REVIEW): 'ON REVIEW',
        str(sly.api.labeling_job_api.LabelingJobApi.Status.COMPLETED): 'COMPLETED',
        str(sly.api.labeling_job_api.LabelingJobApi.Status.STOPPED): 'STOPPED',
    }
    labeling_jobs = sorted(utils.get_labeling_jobs(project.id), key=lambda lj: lj.id, reverse=True)
    lj = labeling_jobs[0]
    started_at = lj.started_at
    status = status_text[lj.status]
    if lj.status == str(sly.api.labeling_job_api.LabelingJobApi.Status.PENDING):
        started_at = "never"

    exam_score = project.custom_data.get("overall_score", None)
    last_lj_id = project.custom_data.get("last_labeling_job_id", None)
    if exam_score is not None and last_lj_id == lj.id:
        if exam_score*100 > passmark:
            status = f"PASSED ({round(exam_score*100, 2)}%)"
        else:
            status = f"FAILED ({round(exam_score*100, 2)}%)"
    
    return {
        "user": project.name[project.name.find(". User: ")+8:],
        "try": len(labeling_jobs),
        "started": started_at,
        "status": status,
        "report": {"workspace_id": workspace_id, "project_id": project.id},
        "loading": False,
        "passmark": passmark,
    }


def generate_table_row_data(exam):
    workspace = exam["workspace_info"]    
    exam_users_projects = exam["exam_projects"]
    exam_user_project_meta = exam["exam_project_meta"]
    exam_benchmark_project = exam["benchmark_project"]
    exam_benchmark_project_meta = exam["benchmark_project_meta"]
    return {
        "exam": workspace.name[7:-1],
        "passmark": exam_benchmark_project.custom_data["passmark"],
        "attempts": "∞" if exam_benchmark_project.custom_data["attempts"] is None else exam_benchmark_project.custom_data["attempts"],
        "classes": {
            "gt": [{"class_name": oc.name, "color": utils.rgb_to_hex(*oc.color)} for oc in exam_user_project_meta.obj_classes], 
            "pred": [{"class_name": oc.name, "color": utils.rgb_to_hex(*oc.color)} for oc in exam_benchmark_project_meta.obj_classes]},
        "tags": {
            "gt": [{"tag_name": tm.name, "color": utils.rgb_to_hex(*tm.color)} for tm in exam_user_project_meta.tag_metas],
            "pred": [{"tag_name": tm.name, "color": utils.rgb_to_hex(*tm.color)} for tm in exam_benchmark_project_meta.tag_metas],
        },
        "created_at": workspace.created_at,
        "assignees": [pr.name for pr in exam_users_projects],
        "benchmark_project": {
            "name": exam_benchmark_project.name,
            "url": exam_benchmark_project.url,
            "preview_url": exam_benchmark_project.image_preview_url,
            "description": f"{exam_benchmark_project.items_count} {exam_benchmark_project.type} in project"
        },
        "created_by": g.api.user.get_info_by_id(exam_benchmark_project.custom_data["created_by"]).name,
        "expandable_content": {
            "table_data": {
                "columns": ExpandableTable.columns.EXAM_USERS_TABLE_COLUMNS,
                "data": [generate_exam_users_table_row_data(
                    project=user_project,
                    workspace_id=workspace.id,
                    passmark=exam_benchmark_project.custom_data["passmark"]
                ) for user_project in exam_users_projects]
            }
        }
    }

def update_exams_table():
    exams_table.loading = True
    exams = g.load_all_exams()
    data = [generate_table_row_data(exam) for exam in exams.values()]
    exams_table.read_json({"columns": ExpandableTable.columns.EXAMS_TABLE_COLUMNS, "data": data})
    exams_table.loading = False


def clean_exams_table():
    exams_table.read_json({"columns": ExpandableTable.columns.EXAMS_TABLE_COLUMNS, "data": []})


new_exam_button = Button(text="New Exam", icon="zmdi zmdi-plus")
refresh_btn = Button(text="Refresh table", icon="zmdi zmdi-refresh")
@refresh_btn.click
def refresh_exams_table():
    update_exams_table()
    

layout = Container(widgets=[Text("<h2>Exams</h2>"), Container(widgets=[refresh_btn, Empty(), new_exam_button], direction="horizontal", fractions=[1,99,1], overflow="wrap"), exams_table])
