import supervisely as sly
from supervisely.app.widgets import Container, Image
from src.ui.dashboard import layout as dashboard_layout, exams_table, new_exam_button, update_exams_table
from src.ui.create_exam import layout as create_exam_layout, cancel_btn as create_exam_cancel_btn, create_btn as create_exam_confirm_btn, create_exam, clean_up as clean_up_create_exam
from src.ui.report import layout as results_layout, return_button as report_return_btn, render_report, calculate_report, clean_up as clean_up_report, results as report_results
import src.utils as utils
import src.globals as g


@exams_table.view_clicked
def go_to_report(value_dict):
    workspace_id = value_dict["report"]["workspace_id"]
    project_id = value_dict["report"]["project_id"]

    dashboard_layout.hide()
    create_exam_layout.hide()
    results_layout.show()
    report_results.loading = True

    report = utils.get_report(workspace_id, project_id)
    if report is None:
        report = refresh_report(value_dict)
    
    pred_project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(project_id))
    iou_threshold = g.exams[workspace_id]["benchmark_project"].custom_data["threshold"]
    benchmark_dataset_id = g.exams[workspace_id]["benchmark_dataset"].id
    exam_dataset_id = utils.get_dataset_by_project_id(project_id).id

    render_report(
        report=report,
        benchmark_dataset_id=benchmark_dataset_id,
        exam_dataset_id=exam_dataset_id,
        classes=[oc.name for oc in pred_project_meta.obj_classes],
        tags=[tm.name for tm in pred_project_meta.tag_metas],
        passmark=value_dict["passmark"],
        iou_threshold=iou_threshold
    )
    report_results.loading = False


@exams_table.refresh_clicked
def refresh_report_clicked(value_dict):
    refresh_report(value_dict)

def refresh_report(value_dict):
    g.is_refreshing_report = True

    workspace_id = value_dict["report"]["workspace_id"]
    project_id = value_dict["report"]["project_id"]

    benchmark_dataset = g.exams[workspace_id]["benchmark_dataset"]
    exam_dataset = utils.get_dataset_by_project_id(project_id)
    iou_threshold = g.exams[workspace_id]["benchmark_project"].custom_data["threshold"]
    pred_project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(project_id))

    report = calculate_report(
        benchmark_dataset,
        exam_dataset,
        classes_whitelist=[oc.name for oc in pred_project_meta.obj_classes],
        tags_whitelist=[tm.name for tm in pred_project_meta.tag_metas],
        obj_tags_whitelist=[tm.name for tm in pred_project_meta.tag_metas],
        iou_threshold=iou_threshold,
    )

    utils.save_report(report, workspace_id, project_id)
    utils.update_report_status(report, project_id)
    g.is_refreshing_report = False

    return report


@new_exam_button.click
def go_to_create_exam():
    dashboard_layout.hide()
    results_layout.hide()
    create_exam_layout.show()

@create_exam_cancel_btn.click
def go_to_dashboard():
    results_layout.hide()
    create_exam_layout.hide()
    dashboard_layout.show()
    update_exams_table()
    clean_up_create_exam()

@create_exam_confirm_btn.click
def create_exam_and_return():
    if create_exam():
        go_to_dashboard()

@report_return_btn.click
def return_from_report_to_dashboard():
    go_to_dashboard()
    clean_up_report()

def load_dashboard():
    create_exam_layout.hide()
    results_layout.hide()
    dashboard_layout.show()
    update_exams_table()

layout = Container(widgets=[dashboard_layout, create_exam_layout, results_layout])
