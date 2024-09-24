from supervisely.app.widgets import OneOf, Select
from supervisely.app import show_dialog
import src.globals as g
from src.ui.dashboard import (
    layout as dashboard_layout,
    exams_table,
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
from src.exam import Exam


DASHBOARD = "dashboard"
CREATE_EXAM = "create_exam"
REPORT = "report"

pages = [
    Select.Item(value=DASHBOARD, label="Dashboard", content=dashboard_layout),
    Select.Item(value=CREATE_EXAM, label="Create Exam", content=create_exam_layout),
    Select.Item(value=REPORT, label="Report", content=report_layout),
]
select_page = Select(items=pages)
current_page = OneOf(select_page)


def load_dashboard():
    select_page.set_value(DASHBOARD)
    update_exams_table()


@exams_table.table.view_clicked
def go_to_report(value_dict):
    report_return_btn.disable()
    workspace_id = value_dict["workspace_id"]
    project_id = value_dict["project_id"]
    user_id = value_dict["user_id"]
    exam = g.exams[workspace_id]
    user = exam.get_user(user_id)
    if user is None:
        show_dialog(
            "User not found",
            f"User with id {user_id} not found in the team",
            "error",
        )
        return

    select_page.set_value(REPORT)
    report_results.loading = True

    report = get_report(workspace_id, project_id)
    if report is None:
        report = refresh_report(value_dict)

    attempt = user.get_last_attempt()
    
    render_report(
        report,
        exam,
        user,
        attempt
    )
    report_results.loading = False
    report_return_btn.enable()


@exams_table.table.new_attempt_clicked
def start_new_attempt(value_dict):
    exams_table.table.loading = True
    exams_table.header.loading = True
    workspace_id = value_dict["workspace_id"]
    user_id = value_dict["user_id"]
    attempts = value_dict["try"]
    exam = g.exams[workspace_id]
    exam: Exam
    max_attempts = exam.max_attempts()
    if max_attempts is not None and attempts >= max_attempts:
        return
    user = exam.get_user(user_id)
    if user is None:
        show_dialog(
            "User not found",
            f"User with id {user_id} not found in the team",
            "error",
        )
        return
    attempt = user.get_last_attempt()
    create_attempt(
        workspace=exam.workspace,
        user_id=user_id,
        benchmark_project_meta=exam.benchmark_project_meta,
        benchmark_dataset=attempt.dataset,
        classes=[oc.name for oc in attempt.project_meta.obj_classes],
        tags=[tm.name for tm in attempt.project_meta.tag_metas],
        guide=attempt.labeling_job.readme,
        reviewer=exam.reviewer_id(),
        attempt_num=attempt.number() + 1,
    )
    delete_attempt(attempt)
    update_exams_table()
    exams_table.header.loading = False
    exams_table.table.loading = False


@exams_table.new_exam_button.click
def go_to_create_exam():
    select_page.set_value(CREATE_EXAM)


def go_to_dashboard():
    select_page.set_value(DASHBOARD)
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


layout = current_page
