from typing import List
from supervisely.app.widgets import (
    Text,
    Container,
    Input,
    Field,
    Card,
    Select,
    SelectDataset,
    Flexbox,
    Button,
    InputNumber,
    RadioGroup,
    OneOf,
    Empty,
    Checkbox,
    Progress,
    Editor,
    FileStorageUpload,
    TeamFilesSelector,
)
import supervisely as sly

import src.globals as g
from src.exam import Exam
from src.constants import DEFAULT_EXAM_PASSMARK, DEFAULT_MATCHING_THRESHOLD


title_input = Input(size="small")
exam_title = Field(title="Exam Title", content=title_input)
select_dataset = SelectDataset(size="small", allowed_project_types=g.ALLOWED_MODALITIES)
select_classes = Select(items=[], multiple=True, size="small")
select_all_classes_btn = Button(text="Select all", button_type="text", button_size="mini")
deselect_all_classes_btn = Button(text="Deselect all", button_type="text", button_size="mini")
select_tags = Select(items=[], multiple=True, size="small")
select_all_tags_btn = Button(text="Select all", button_type="text", button_size="mini")
deselect_all_tags_btn = Button(text="Deselect all", button_type="text", button_size="mini")
benchmark_project = Card(
    title="BENCHMARK PROJECT",
    description="Choose perfectly labeled project. Assigned users won't see original labels.\nYou can select only specific classes or tags to be labeled.",
    content=Container(
        widgets=[
            select_dataset,
            Field(
                title="Classes",
                content=Container(
                    widgets=[
                        Flexbox(widgets=[select_all_classes_btn, deselect_all_classes_btn]),
                        select_classes,
                    ],
                    gap=0,
                ),
            ),
            Field(
                title="Tags",
                content=Container(
                    widgets=[
                        Flexbox(widgets=[select_all_tags_btn, deselect_all_tags_btn]),
                        select_tags,
                    ],
                    gap=0,
                ),
            ),
        ],
        gap=20,
    ),
)
select_users = Select(
    items=[Select.Item(user.id, user.login) for user in g.users.values()],
    multiple=True,
    filterable=True,
    size="small",
)
assigned_users = Card(
    title="ASSIGNED USERS",
    description="This exam will appear as Labeling job at the Labeling Jobs page for every user, selected below.",
    content=Field(title="Assign to", content=select_users),
)
select_reviewer = Select(
    items=[Select.Item(user.id, user.login) for user in g.users.values()],
    filterable=True,
    size="small",
)
reviewers = Card(
    title="ASSIGNED REVIEWERS",
    description="Assign reviewers for lableling jobs, created by this exam.",
    content=Field(title="Assign to", content=select_reviewer),
)
guide_editor = Editor(language_mode="plain_text")
guide_file = FileStorageUpload(g.team_id, f"exam_data/guide")
guide_from_team_files = TeamFilesSelector(g.team_id, selection_file_type="file")
exam_guide_select = Select(
    items=[
        Select.Item("none", "None", Empty()),
        Select.Item(
            "editor",
            "Markdown text",
            Field(content=guide_editor, title="Enter Markdown guide as text"),
        ),
        Select.Item(
            "file",
            "Upload Markdown file",
            Field(content=guide_file, title="Upload Markdown guide as file"),
        ),
        Select.Item(
            "team_file",
            "Select Markdown file from Team Files",
            Field(
                content=guide_from_team_files,
                title="Select Markdown guide from Team Files",
            ),
        ),
    ]
)
exam_guide_one_of = OneOf(exam_guide_select)
input_passmark = InputNumber(min=0, max=100, step=1, value=DEFAULT_EXAM_PASSMARK, size="small")
matching_threshold = InputNumber(
    min=0, max=100, step=1, value=DEFAULT_MATCHING_THRESHOLD, size="small"
)
input_attempts = InputNumber(min=0, max=100, step=1, value=1, size="small")
maximum_attempts = RadioGroup(
    items=[
        RadioGroup.Item("Unlimited", content=Empty()),
        RadioGroup.Item("Fixed", content=input_attempts),
    ]
)
segmentation_mode_checkmark = Checkbox(content="Segmentation mode")
time_limit = Select(items=[Select.Item("Unlimited")])
# disable because it is coming soon
time_limit.disable()
show_report_to_labelers_checkmark = Checkbox(content="Show report to labelers")
show_report_to_labelers_checkmark.hide()
exam_settings = Card(
    title="EXAM SETTINGS",
    description="Attach training materials, choose desired pass mark, number of attempts a user can try and other options",
    content=Container(
        widgets=[
            Field(
                title="Guide",
                description="It will be added to labeling job readme",
                content=Container(widgets=[exam_guide_select, exam_guide_one_of]),
            ),
            Field(
                title="Pass mark",
                description="Minimum exam score (in %)",
                content=input_passmark,
            ),
            Field(
                title="Matching threshold",
                description="IoU threshold for matching (in %)",
                content=matching_threshold,
            ),
            Field(
                title="Segmentation mode",
                description='If enabled, geometries of type "Bitmap" and "Polygon" will be treated as segmentation. Label that was added later will overlap older labels.',
                content=segmentation_mode_checkmark,
            ),
            Field(
                title="Maximum Attempts",
                description="How many times labeler could try to pass this exam",
                content=Container(widgets=[maximum_attempts, OneOf(maximum_attempts)]),
            ),
            Field(
                title="Time limit",
                description="Time limit in hours (coming soon)",
                content=time_limit,
            ),
            Field(title="", content=show_report_to_labelers_checkmark),
        ]
    ),
)
create_btn = Button(text="Create")
status_bar = Progress(message="Creating Exam...", show_percents=True)
status_bar.hide()


def get_guide():
    val = exam_guide_select.get_value()
    if val == "none":
        return ""
    if val == "editor":
        return guide_editor.get_text()
    if val == "file":
        paths = guide_file.get_uploaded_paths()
        if len(paths) == 0:
            return None
        path = paths[0]
        g.api.file.download(g.team_id, path, "guide.md")
        with open("guide.md", "r") as f:
            return f.read()
    if val == "team_file":
        paths = guide_from_team_files.get_selected_paths()
        if len(paths) == 0:
            return None
        path = paths[0]
        g.api.file.download(g.team_id, path, "guide.md")
        with open("guide.md", "r") as f:
            return f.read()


def create_exam_workspace(exam_name: str):
    ws = g.api.workspace.create(
        team_id=g.team_id,
        name=f'Exam: "{exam_name}"',
        description="",
        change_name_if_conflict=False,
    )
    return ws


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
    if project.type == str(sly.ProjectType.IMAGES):
        items_ids = [img.id for img in g.api.image.get_list(dataset.id)]
    elif project.type == str(sly.ProjectType.VIDEOS):
        items_ids = [vid.id for vid in g.api.video.get_list(dataset.id)]
    else:
        raise ValueError("Invalid project type")
    return g.api.labeling_job.create(
        name=project.name,
        dataset_id=dataset.id,
        user_ids=user_ids,
        readme=readme,
        description=description,
        classes_to_label=classes_to_label,
        tags_to_label=tags_to_label,
        reviewer_id=reviewer_id,
        images_ids=items_ids,
    )


def create_attempt_project_for_user(
    workspace: sly.api.workspace_api.WorkspaceInfo,
    benchmark_project: sly.ProjectInfo,
    user: sly.api.user_api.UserInfo,
    attempt: int,
    reviewer: int,
):
    project = g.api.project.create(
        workspace_id=workspace.id,
        name=f"{workspace.name}. User: {user.login}. Attempt: {attempt}",
        type=benchmark_project.type,
        description="",
        change_name_if_conflict=True,
    )
    project = g.api.project.get_info_by_id(project.id)
    custom_data = project.custom_data
    custom_data["is_attempt_project"] = True
    custom_data["user_id"] = user.id
    custom_data["user_name"] = user.name
    custom_data["user_login"] = user.login
    custom_data["attempt"] = attempt
    custom_data["reviewer_id"] = reviewer
    g.api.project.update_custom_data(project.id, custom_data)
    return project


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


def create_attempt(
    workspace,
    user_id,
    benchmark_project,
    benchmark_project_meta,
    benchmark_dataset,
    classes,
    tags,
    guide,
    reviewer,
    attempt_num,
):
    user_info = g.users.get(user_id)
    attempt_project = create_attempt_project_for_user(
        workspace, benchmark_project, user_info, attempt_num, reviewer
    )
    attempt_project_meta = create_project_meta(benchmark_project_meta, classes, tags)
    g.api.project.update_meta(attempt_project.id, attempt_project_meta)
    attempt_dataset = g.api.dataset.copy(
        attempt_project.id, benchmark_dataset.id, new_name=benchmark_dataset.name
    )
    labeling_job = create_labeling_job(
        project=attempt_project,
        dataset=attempt_dataset,
        user_ids=[user_id],
        readme=guide,
        description="",
        classes_to_label=classes,
        tags_to_label=tags,
        reviewer_id=reviewer,
    )


def delete_attempt(attempt: Exam.ExamUser.Attempt):
    g.api.labeling_job.archive(attempt.labeling_job.id)


def create_exam():
    exam_name = title_input.get_value()
    if exam_name.isspace() or exam_name == "":
        sly.app.show_dialog(
            "Name not specified",
            "Name not specified. Please provide a name for the Exam",
            status="warning",
        )
        return False
    source_dataset_id = select_dataset.get_selected_id()
    classes_whitelist = select_classes.get_value()
    tags_whitelist = select_tags.get_value()
    users = select_users.get_value()
    reviewer = select_reviewer.get_value()
    guide = get_guide()
    if guide is None:
        sly.app.show_dialog(
            "Guide not selected",
            "Guide not selected. Please select file",
            status="warning",
        )
        return False
    passmark = input_passmark.get_value()
    threshold = matching_threshold.get_value()
    attempts = None if maximum_attempts.get_value() == "Unlimited" else input_attempts.get_value()
    show_report_to_labelers = show_report_to_labelers_checkmark.is_checked()
    segmentation_mode = segmentation_mode_checkmark.is_checked()

    if g.api.workspace.exists(g.team_id, f'Exam: "{exam_name}"'):
        sly.app.show_dialog(
            "Exam exist",
            f'Exam with name "{exam_name}" already exists. Please enter new name',
            "warning",
        )
        return False

    if not source_dataset_id:
        sly.app.show_dialog(
            "Dataset not selected",
            f"Dataset not selected. Please select dataset",
            "warning",
        )
        return False

    if len(users) == 0:
        sly.app.show_dialog(
            "Users not selected",
            f"Users not selected. Please select at lease one user",
            "warning",
        )
        return False

    status_bar.show()
    progress = status_bar(iterable=[step for step in range(100)])

    # create workspace
    exam_workspace = create_exam_workspace(exam_name)
    progress.update(100 // (2 + len(users)))

    # copy benchmark project to exam worksapce
    source_dataset_info = g.api.dataset.get_info_by_id(source_dataset_id)
    benchmark_project_meta = sly.ProjectMeta.from_json(
        g.api.project.get_meta(source_dataset_info.project_id)
    )
    copy_project_task_id = g.api.project.clone(
        source_dataset_info.project_id,
        exam_workspace.id,
        f"{exam_workspace.name}. Benchmark project",
    )
    g.api.task.wait(copy_project_task_id, g.api.task.Status.FINISHED)

    benchmark_project = g.api.project.get_info_by_name(
        exam_workspace.id, f"{exam_workspace.name}. Benchmark project"
    )
    benchmark_dataset = g.api.dataset.get_info_by_name(
        benchmark_project.id, source_dataset_info.name
    )

    # add exam settings to becnhmark project custom data
    benchmark_project_custom_data = {
        "is_benchmark_project": True,
        "source_project_id": source_dataset_info.project_id,
        "source_dataset_id": source_dataset_id,
        "benchmark_dataset_id": benchmark_dataset.id,
        "exam_name": exam_name,
        "classes": classes_whitelist,
        "tags": tags_whitelist,
        "assignees": users,
        "passmark": passmark,
        "threshold": threshold,
        "attempts": attempts,
        "show_report_to_labelers": show_report_to_labelers,
        "created_by": g.user_id,
        "segmentation_mode": segmentation_mode,
        "reviewer": reviewer,
    }
    g.api.project.update_custom_data(benchmark_project.id, benchmark_project_custom_data)
    progress.update(100 // (2 + len(users)))

    # create project and labeling job for each user
    for user_id in users:
        create_attempt(
            workspace=exam_workspace,
            user_id=user_id,
            benchmark_project=benchmark_project,
            benchmark_project_meta=benchmark_project_meta,
            benchmark_dataset=benchmark_dataset,
            classes=classes_whitelist,
            tags=tags_whitelist,
            guide=guide,
            reviewer=reviewer,
            attempt_num=1,
        )
        progress.update(100 // (2 + len(users)))
    status_bar.hide()
    return True


cancel_btn = Button(text="Cancel")
confirm_buttons = Flexbox(widgets=[create_btn, cancel_btn])
return_btn = Button(text="Return to Exams", button_size="small", icon="zmdi zmdi-arrow-left")


def clean_up():
    title_input.set_value("")
    select_classes.set_value([])
    select_tags.set_value([])
    select_users.set_value([])
    guide_editor.set_text("")
    input_passmark.value = DEFAULT_EXAM_PASSMARK
    matching_threshold.value = DEFAULT_MATCHING_THRESHOLD


@select_dataset.value_changed
def selected_dataset(value):
    # value - dataset_id
    if value is None:
        select_classes.disable()
        select_tags.disable()
        return
    selected_dataset = g.api.dataset.get_info_by_id(value)
    classes = sly.ProjectMeta.from_json(
        g.api.project.get_meta(selected_dataset.project_id)
    ).obj_classes
    tags = sly.ProjectMeta.from_json(g.api.project.get_meta(selected_dataset.project_id)).tag_metas
    select_classes.set(items=[Select.Item(obj_class.name, obj_class.name) for obj_class in classes])
    select_all_classes()
    select_classes.enable()
    select_tags.set(items=[Select.Item(tag_meta.name, tag_meta.name) for tag_meta in tags])
    select_all_tags()
    select_tags.enable()


@select_all_classes_btn.click
def select_all_classes():
    select_classes.set_value([item.value for item in select_classes.get_items()])


@deselect_all_classes_btn.click
def deselect_all_classes():
    select_classes.set_value([])


@select_all_tags_btn.click
def select_all_tags():
    select_tags.set_value([item.value for item in select_tags.get_items()])


@deselect_all_tags_btn.click
def deselect_all_tags():
    select_tags.set_value([])


layout = Container(
    widgets=[
        Text("<h2>Create Exam</h2>"),
        return_btn,
        exam_title,
        benchmark_project,
        assigned_users,
        reviewers,
        exam_settings,
        confirm_buttons,
        status_bar,
    ]
)
