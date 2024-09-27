from supervisely.app.widgets import OneOf, Select, Container, Progress
import supervisely as sly

import src.report.image_report as image_report
import src.report.video_report as video_report
from src.report.report import (
    return_button,
    error_notification,
    get_report,
    save_report,
    error_in_report,
)
from src.exam import Exam
import src.globals as g


results_select = Select(
    items=[
        Select.Item(value="image_results", content=image_report.results),
        Select.Item(value="video_results", content=video_report.results),
    ]
)
progress = Progress("Calculating report...", show_percents=True)
progress.hide()
results = OneOf(results_select)
layout = Container(widgets=[return_button, progress, results, error_notification], gap=20)


def calculate_report(exam: Exam, attempt: Exam.ExamUser.Attempt, progress):
    if exam.benchmark_project.type == str(sly.ProjectType.IMAGES):
        return image_report.calculate_report(exam, attempt, progress)
    elif exam.benchmark_project.type == str(sly.ProjectType.VIDEOS):
        return video_report.calculate_report(exam, attempt, progress)
    else:
        raise RuntimeError("Unknown project type.")


def update_report_status(report, attempt: Exam.ExamUser.Attempt):
    if attempt.project.type == str(sly.ProjectType.IMAGES):
        get_overall_score = image_report.get_overall_score
    elif attempt.project.type == str(sly.ProjectType.VIDEOS):

        def get_overall_score(report):
            report = video_report.report_to_dict(report)
            return video_report.get_overall_score(report)

    else:
        raise RuntimeError("Unknown project type.")

    custom_data = g.api.project.get_info_by_id(attempt.project.id).custom_data
    if error_in_report(report):
        custom_data["overall_score"] = "Error"
    else:
        custom_data["overall_score"] = get_overall_score(report)
    custom_data = g.api.project.get_info_by_id(attempt.project.id).custom_data
    g.api.project.update_custom_data(attempt.project.id, custom_data)


def refresh_report(value_dict):
    g.is_refreshing_report = True

    workspace_id = value_dict["workspace_id"]
    user_id = value_dict["user_id"]

    exam: Exam = g.exams[workspace_id]
    attempt = exam.get_user(user_id).get_last_attempt()

    report = calculate_report(
        exam=exam,
        attempt=attempt,
        progress=progress,
    )

    save_report(report, attempt, progress)
    update_report_status(report, attempt)
    g.is_refreshing_report = False

    return report


def render_report(
    report,
    exam: Exam,
    user: Exam.ExamUser,
    attempt: Exam.ExamUser.Attempt,
):
    if exam.benchmark_project.type == str(sly.ProjectType.IMAGES):
        results_select.set_value("image_results")
        return image_report.render_report(report, exam, user, attempt, progress)
    elif exam.benchmark_project.type == str(sly.ProjectType.VIDEOS):
        results_select.set_value("video_results")
        return video_report.render_report(report, exam, user, attempt, progress)
    else:
        raise RuntimeError("Unknown project type.")


def clean_up():
    image_report.clean_up()
    video_report.clean_up()
