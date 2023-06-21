import copy
import dateutil.parser
from supervisely.app.widgets import (
    Text,
    Container,
    Button,
    Input,
    Select,
    Card,
    Flexbox,
)
from supervisely.api.labeling_job_api import LabelingJobApi

import src.globals as g
from src.ui.expandable_table import ExpandableTable
from src.ui.report import calculate_report, save_report, update_report_status


def get_user_login(user: g.Exam.ExamUser):
    try:
        return user.attempts[0].project.custom_data.get["user_login"]
    except:
        return g.api.user.get_info_by_id(user.user_id).login
    
def get_user_name(user: g.Exam.ExamUser):
    try:
        return user.attempts[0].project.custom_data.get["user_name"]
    except:
        return g.api.user.get_info_by_id(user.user_id).name

class ExamsTable:
    def __init__(self, expandable_table: ExpandableTable, header: Container):
        self.table = expandable_table
        self.header = header
        self.exams_rows = []
        self.applied_filters = {}

        self._available_filters = {
            "assignee": ExamsTable.filter_by_assignee,
            "creator": ExamsTable.filter_by_creator,
            "status": ExamsTable.filter_by_status,
        }

    def update(self):
        self.table.loading = True
        self.header.loading = True
        exams = g.load_all_exams()
        self.exams_rows = [
            ExpandableTable.Exam(
                name=exam.workspace.name[7:-1],
                workspace=exam.workspace,
                benchmark_project=exam.benchmark_project,
                benchmark_project_meta=exam.benchmark_project_meta,
                passmark=exam.benchmark_project.custom_data["passmark"],
                created_at=exam.workspace.created_at,
                created_by=g.api.user.get_info_by_id(
                    exam.benchmark_project.custom_data["created_by"]
                ),
                attempts=exam.benchmark_project.custom_data["attempts"],
                classes=exam.exam_project_meta.obj_classes,
                tags=exam.exam_project_meta.tag_metas,
                users=[
                    ExpandableTable.ExamUser(
                        user_name=get_user_name(user),
                        user_id=user.user_id,
                        user_login=get_user_login(user),
                        attempts=len(user.attempts),
                        exam_project=user.attempts[0].project,
                        labeling_job=user.attempts[0].labeling_job,
                    )
                    for user in exam.users.values()
                ],
            )
            for exam in exams.values()
        ]
        self.table.set(self.exams_rows)

        all_users = []
        creators = []
        for exam in self.exams_rows:
            for user in exam._users:
                if user._user_id not in [u[0] for u in all_users]:
                    all_users.append((user._user_id, user._user_name))
            if exam._created_by.id not in [c[0] for c in creators]:
                creators.append((exam._created_by.id, exam._created_by.name))
        select_filter_by_assignee.set(
            items=[Select.Item(user[0], user[1]) for user in all_users]
        )
        select_filter_by_assignee.set_value([])
        select_filter_by_creator.set(
            items=[Select.Item(creator[0], creator[1]) for creator in creators]
        )
        select_filter_by_creator.set_value([])

        self.table.loading = False
        self.header.loading = False

    def search_by_name(self, text: str):
        self.header.loading = True
        self.table.loading = True
        exams = [exam for exam in self.exams_rows if exam.get_name().find(text) != -1]
        self.table.set(exams)
        self.table.loading = False
        self.header.loading = False

    def filter(self):
        self.header.loading = True
        self.table.loading = True
        exams = self.exams_rows.copy()
        for filter_name, filter_val in self.applied_filters.items():
            exams = self._available_filters[filter_name](exams, filter_val)
        self.table.set(exams)
        self.header.loading = False
        self.table.loading = False

    def sort(self, val):
        self.header.loading = True
        self.table.loading = True

        key = lambda exam: dateutil.parser.isoparse(exam._created_at.rstrip("Z"))
        self.table.set(sorted(self.table._exams, key=key, reverse=val == "oldest"))

        self.header.loading = False
        self.table.loading = False

    def filter_changed(self, key, val):
        if type(val) != list:
            return
        self.applied_filters[key] = val
        self.filter()

    @staticmethod
    def filter_by_assignee(exams, filter_val):
        if len(filter_val) == 0:
            return exams
        filtered_exams = []
        for exam in exams:
            users = []
            for user in exam._users:
                if user._user_id in filter_val:
                    users.append(user)
            if len(users) != 0:
                new_exam = copy.deepcopy(exam)
                new_exam._users = users
                filtered_exams.append(new_exam)
        return filtered_exams

    @staticmethod
    def filter_by_creator(exams, filter_val):
        if len(filter_val) == 0:
            return exams
        filtered_exams = []
        for exam in exams:
            if exam._created_by.id in filter_val:
                filtered_exams.append(exam)
        return filtered_exams

    @staticmethod
    def filter_by_status(exams, filter_val):
        if len(filter_val) == 0:
            return exams
        filtered_exams = []
        for exam in exams:
            users = []
            for user in exam._users:
                labeling_job = user._labeling_job
                status = labeling_job.status
                exam_score = user._exam_project.custom_data.get("overall_score", None)
                if exam_score is not None:
                    if exam_score * 100 > exam._passmark:
                        status = f"passed"
                    else:
                        status = f"failed"

                if status in filter_val:
                    users.append(user)
            if len(users) != 0:
                new_exam = copy.deepcopy(exam)
                new_exam._users = users
                filtered_exams.append(new_exam)
        return filtered_exams


refresh_btn = Button(text="", icon="zmdi zmdi-refresh", button_size="mini", icon_gap=0)
new_exam_button = Button(text="New Exam", icon="zmdi zmdi-plus", button_size="mini")
search_exam = Input(placeholder="Search by title", size="mini")
search_btn = Button("", button_size="mini", icon="zmdi zmdi-search", icon_gap=0)
select_filter_by_assignee = Select(
    placeholder="Assignee", items=[], multiple=True, size="mini"
)
select_filter_by_creator = Select(
    placeholder="Creator", items=[], multiple=True, size="mini"
)
select_filter_by_status = Select(
    items=[
        Select.Item(str(LabelingJobApi.Status.PENDING), "PENDING"),
        Select.Item(str(LabelingJobApi.Status.IN_PROGRESS), "IN PROGRESS"),
        Select.Item(str(LabelingJobApi.Status.ON_REVIEW), "ON REVIEW"),
        Select.Item(str(LabelingJobApi.Status.COMPLETED), "COMPLETED"),
        Select.Item(str(LabelingJobApi.Status.STOPPED), "STOPPED"),
        Select.Item("passed", "PASSED"),
        Select.Item("failed", "FAILED"),
    ],
    placeholder="Status",
    multiple=True,
    size="mini",
)
select_sort = Select(
    items=[Select.Item("newest"), Select.Item("oldest")],
    placeholder="Sort",
    size="mini",
)

table_header = Card(
    title="Filters",
    content=Flexbox(
        widgets=[
            select_filter_by_assignee,
            select_filter_by_creator,
            select_filter_by_status,
            select_sort,
        ]
    ),
    content_top_right=Container(
        widgets=[search_exam, search_btn, new_exam_button],
        direction="horizontal",
        fractions=[7, 1, 4],
        gap=5,
        overflow="wrap",
    ),
)
exams_table = ExamsTable(ExpandableTable([]), table_header)
exams_table_card = Card(
    title="Exams", content=exams_table.table, content_top_right=refresh_btn
)
table_and_header = Container(widgets=[table_header, exams_table_card])


def update_exams_table():
    exams_table.update()


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


exams_table.table.refresh_clicked(refresh_report)


@select_filter_by_assignee.value_changed
def filter_by_assignee_func(val):
    exams_table.filter_changed("assignee", val)


@select_filter_by_creator.value_changed
def filter_by_creator_func(val):
    exams_table.filter_changed("creator", val)


@select_filter_by_status.value_changed
def filter_by_status_func(val):
    exams_table.filter_changed("status", val)


@select_sort.value_changed
def sort_exams_table(val):
    exams_table.sort(val)


@search_btn.click
def filter_table_by_name():
    val = search_exam.get_value()
    exams_table.search_by_name(val)


refresh_btn.click(update_exams_table)

layout = Container(widgets=[Text("<h2>Exams</h2>"), table_and_header])
