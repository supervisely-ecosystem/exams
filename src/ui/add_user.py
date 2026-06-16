from supervisely.app import show_dialog
from supervisely.app.widgets import Button, Card, Container, Field, Select, Text

import src.globals as g
from src.exam import Exam
from src.ui.create_exam import create_attempt


_current_workspace_id = None
_title = Text("<h2>Add User to Exam</h2>")
_exam_name = Text("")
_select_user = Select(items=[], filterable=True, size="small")
_assign_card = Card(
    title="ASSIGN USER",
    description="Select a team member who should receive the first attempt in this exam.",
    content=Field(title="User", content=_select_user),
)
confirm_btn = Button(text="Add User")
cancel_btn = Button(text="Cancel")
return_btn = Button(text="Return to Exams", button_size="small", icon="zmdi zmdi-arrow-left")
layout = Container(
    widgets=[
        _title,
        return_btn,
        _exam_name,
        _assign_card,
        Container(widgets=[confirm_btn, cancel_btn], direction="horizontal", gap=10),
    ]
)


def _get_exam() -> Exam:
    if _current_workspace_id is None:
        raise RuntimeError("Exam is not selected.")
    return g.exams[_current_workspace_id]


def _get_available_users(exam: Exam):
    assigned_user_ids = {user.user_id for user in exam.get_all_users()}
    users = [user for user in g.users.values() if user.id not in assigned_user_ids]
    users.sort(key=lambda user: (user.name or user.login or "").lower())
    return users


def _get_guide(exam: Exam):
    for user in exam.get_all_users():
        attempt = user.get_last_attempt()
        readme = getattr(attempt.labeling_job, "readme", None)
        if readme:
            return readme
    return ""


def open_page(workspace_id: int):
    global _current_workspace_id
    _current_workspace_id = workspace_id
    exam = g.exams[workspace_id]
    available_users = _get_available_users(exam)
    _exam_name.set(f"<b>{exam.name()}</b>", status="text")
    _select_user.set(
        items=[
            Select.Item(user.id, user.name or user.login or f"User {user.id}")
            for user in available_users
        ]
    )
    _select_user.set_value(None)
    if len(available_users) == 0:
        _select_user.disable()
    else:
        _select_user.enable()


def clean_up():
    global _current_workspace_id
    _current_workspace_id = None
    _select_user.set_value(None)


def add_user_to_exam():
    try:
        if _current_workspace_id is None:
            show_dialog("Exam not selected", "Open an exam before adding a user.", "warning")
            return False

        exam = _get_exam()
        user_id = _select_user.get_value()
        if user_id is None:
            show_dialog("User not selected", "Select a user to add to the exam.", "warning")
            return False

        if user_id in [user.user_id for user in exam.get_all_users()]:
            show_dialog("User already assigned", "This user already has an attempt in the exam.", "warning")
            return False

        guide = _get_guide(exam)
        create_attempt(
            workspace=exam.workspace,
            user_id=user_id,
            benchmark_project=exam.benchmark_project,
            benchmark_project_meta=exam.benchmark_project_meta,
            benchmark_dataset=exam.benchmark_dataset,
            classes=[obj_class.name for obj_class in exam.attempt_project_meta.obj_classes],
            tags=[tag_meta.name for tag_meta in exam.attempt_project_meta.tag_metas],
            guide=guide,
            reviewer=exam.reviewer_id(),
            attempt_num=1,
        )

        custom_data = exam.benchmark_project.custom_data
        assignees = custom_data.get("assignees", [])
        if user_id not in assignees:
            assignees.append(user_id)
            custom_data["assignees"] = assignees
            g.api.project.update_custom_data(exam.benchmark_project.id, custom_data)

        clean_up()
        return True
    except Exception as e:
        show_dialog("Failed to add user", str(e), "error")
        return False
