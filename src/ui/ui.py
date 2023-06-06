from supervisely.app.widgets import Container
import supervisely as sly
import src.globals as g
from src.ui.dashboard import (
    layout as dashboard_layout,
    exams_table,
    new_exam_button,
    update_exams_table,
)
from src.ui.create_exam import (
    layout as create_exam_layout,
    return_btn as create_exam_return_btn,
    cancel_btn as create_exam_cancel_btn,
    create_btn as create_exam_confirm_btn,
    create_exam,
    create_attempt,
    delete_attempt,
    clean_up as create_exam_clean_up,
)
from src.ui.report import (
    layout as report_layout,
    return_button as report_return_btn,
    results as report_results,
    get_report,
    refresh_report,
    render_report,
    clean_up as report_clean_up,
)


def load_dashboard():
    create_exam_layout.hide()
    report_layout.hide()
    dashboard_layout.show()
    update_exams_table()


@exams_table.table.view_clicked
def go_to_report(value_dict):
    report_return_btn.disable()
    workspace_id = value_dict["workspace_id"]
    project_id = value_dict["project_id"]
    user_id = value_dict["user_id"]

    dashboard_layout.hide()
    create_exam_layout.hide()
    report_layout.show()
    report_results.loading = True

    report = get_report(workspace_id, project_id)
    if report is None:
        report = refresh_report(value_dict)

    pred_project_meta = g.exams[workspace_id].users[user_id].attempts[0].project_meta
    iou_threshold = g.exams[workspace_id].benchmark_project.custom_data["threshold"]
    benchmark_project = g.exams[workspace_id].benchmark_project
    benchmark_dataset = g.exams[workspace_id].benchmark_dataset
    exam_project = g.exams[workspace_id].users[user_id].attempts[0].project
    exam_dataset = g.exams[workspace_id].users[user_id].attempts[0].dataset
    user_name = exam_project.custom_data.get("user_name", None)

    render_report(
        report=report,
        benchmark_project=benchmark_project,
        benchmark_dataset=benchmark_dataset,
        exam_project=exam_project,
        exam_dataset=exam_dataset,
        classes=[oc.name for oc in pred_project_meta.obj_classes],
        tags=[tm.name for tm in pred_project_meta.tag_metas],
        passmark=value_dict["passmark"],
        iou_threshold=iou_threshold,
        user_name=user_name,
    )
    report_results.loading = False
    report_return_btn.enable()


@exams_table.table.new_attempt_clicked
def start_new_attempt(value_dict):
    exams_table.table.loading = True
    workspace_id = value_dict["workspace_id"]
    user_id = value_dict["user_id"]
    exam = g.exams[workspace_id]
    attempts = value_dict["try"]
    max_attempts = exam.benchmark_project.custom_data["attempts"]
    if max_attempts is not None and attempts >= max_attempts:
        return
    attempt = exam.users[user_id].attempts[0]
    delete_attempt(attempt)
    create_attempt(
        workspace=exam.workspace,
        user_id=user_id,
        benchmark_project_meta=exam.benchmark_project_meta,
        benchmark_dataset=attempt.dataset,
        classes=[oc.name for oc in exam.benchmark_project_meta.obj_classes],
        tags=[tm.name for tm in exam.benchmark_project_meta.tag_metas],
        guide=attempt.labeling_job.readme,
        reviewer=attempt.project.custom_data["reviewer_id"],
        attempt_num=attempt.project.custom_data["attempt"] + 1,
    )
    update_exams_table()
    exams_table.table.loading = False


@new_exam_button.click
def go_to_create_exam():
    dashboard_layout.hide()
    report_layout.hide()
    create_exam_layout.show()


def go_to_dashboard():
    report_layout.hide()
    create_exam_layout.hide()
    dashboard_layout.show()
    create_exam_clean_up()


create_exam_cancel_btn.click(go_to_dashboard)
create_exam_return_btn.click(go_to_dashboard)


@create_exam_confirm_btn.click
def create_exam_and_return():
    if create_exam():
        go_to_dashboard()
        update_exams_table()


@report_return_btn.click
def return_from_report_to_dashboard():
    go_to_dashboard()
    report_clean_up()


layout = Container(widgets=[dashboard_layout, create_exam_layout, report_layout])
