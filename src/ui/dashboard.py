import copy
import traceback
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
from supervisely import logger, timeit

import src.globals as g
from src.ui.widgets.expandable_table import ExpandableTable
from src.exam import Exam


def get_user_login(exam_user: Exam.ExamUser):
    return g.users.get(exam_user.user_id).login


def get_user_name(exam_user: Exam.ExamUser):
    return g.users.get(exam_user.user_id).name


@timeit
def load_all_exams():
    g.exams = {}
    exam_workspaces = [
        ws
        for ws in g.api.workspace.get_list(
            team_id=g.team_id,
            #filters=[{"field": "name", "operator": "like", "value": "Exam: %"}],
        ) if ws.name.startswith("Exam: ")
    ]
    for workspace in exam_workspaces:
        try:
            exam = Exam.load_from_workspace(workspace, g.api)
            g.exams[workspace.id] = exam
        except:
            logger.info(
                "Failed to load exam from workspace",
                exc_info=traceback.format_exc(),
                extra={
                    "workspace_id": workspace.id,
                    "workspace_name": workspace.name,
                },
            )
    return g.exams


class ExamsTable:
    def __init__(self):
        self._create_table()
        self._create_header()
        self._create_content()

        self.exams_rows = []
        self.applied_filters = {}

        self._available_filters = {
            "assignee": ExamsTable.filter_by_assignee,
            "creator": ExamsTable.filter_by_creator,
            "status": ExamsTable.filter_by_status,
        }

    def _create_table(self):
        self.table = ExpandableTable([])

    def _create_header(self):
        self.new_exam_button = Button(text="New Exam", icon="zmdi zmdi-plus", button_size="mini")
        self._search_by_name = Input(placeholder="Search by title", size="mini")
        search_btn = Button("", button_size="mini", icon="zmdi zmdi-search", icon_gap=0)
        self._select_filter_by_assignee = Select(
            placeholder="Assignee", items=[], multiple=True, size="mini"
        )
        self._select_filter_by_creator = Select(
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

        self.header = Card(
            title="Filters",
            content=Flexbox(
                widgets=[
                    self._select_filter_by_assignee,
                    self._select_filter_by_creator,
                    select_filter_by_status,
                    select_sort,
                ]
            ),
            content_top_right=Container(
                widgets=[self._search_by_name, search_btn, self.new_exam_button],
                direction="horizontal",
                fractions=[7, 1, 4],
                gap=5,
                overflow="wrap",
            ),
        )

        @self._select_filter_by_assignee.value_changed
        def filter_by_assignee_func(val):
            self.filter_changed("assignee", val)

        @self._select_filter_by_creator.value_changed
        def filter_by_creator_func(val):
            self.filter_changed("creator", val)

        @select_filter_by_status.value_changed
        def filter_by_status_func(val):
            self.filter_changed("status", val)

        @select_sort.value_changed
        def sort_exams_table(val):
            self.sort(val)

        @search_btn.click
        def filter_table_by_name():
            val = self._search_by_name.get_value()
            self.search_by_name(val)

    def _create_content(self):
        refresh_btn = Button(
            text="", icon="zmdi zmdi-refresh", button_size="mini", icon_gap=0
        )
        refresh_btn.click(self.refresh)
        self.content = Container(
            widgets=[
                self.header,
                Card(title="Exams", content=self.table, content_top_right=refresh_btn),
            ]
        )

    def refresh(self):
        self.table.loading = True
        self.header.loading = True
        exams = load_all_exams()
        self.exams_rows = [
            ExpandableTable.Exam(
                name=exam.name(),
                workspace=exam.workspace,
                benchmark_project=exam.benchmark_project,
                benchmark_project_meta=exam.benchmark_project_meta,
                passmark=exam.get_passmark(),
                created_at=exam.created_at(),
                created_by=g.users.get(exam.created_by()),
                attempts=exam.max_attempts(),
                classes=exam.attempt_project_meta.obj_classes,
                tags=exam.attempt_project_meta.tag_metas,
                users=[
                    ExpandableTable.ExamUser(
                        user_name=get_user_name(exam_user),
                        user_id=exam_user.user_id,
                        user_login=get_user_login(exam_user),
                        attempts=len(exam_user.attempts),
                        attempt_project=exam_user.get_last_attempt().project,
                        labeling_job=exam_user.get_last_attempt().labeling_job,
                    )
                    for exam_user in exam.get_all_users()
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
        self._select_filter_by_assignee.set(
            items=[Select.Item(user[0], user[1]) for user in all_users]
        )
        self._select_filter_by_assignee.set_value([])
        self._select_filter_by_creator.set(
            items=[Select.Item(creator[0], creator[1]) for creator in creators]
        )
        self._select_filter_by_creator.set_value([])

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
        self.table.set(sorted(self.table._exams, key=key, reverse=val == "newest"))

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
                exam_score = user._attempt_project.custom_data.get("overall_score", None)
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


exams_table = ExamsTable()


def update_exams_table():
    exams_table.refresh()


# def refresh_report(value_dict):
#     g.is_refreshing_report = True

#     workspace_id = value_dict["workspace_id"]
#     user_id = value_dict["user_id"]

#     benchmark_dataset = g.exams[workspace_id].benchmark_dataset
#     attempt = g.exams[workspace_id].users[user_id].attempts[0]
#     iou_threshold = g.exams[workspace_id].benchmark_project.custom_data["threshold"]
#     attempt_project_meta = attempt.project_meta

#     report = calculate_report(
#         benchmark_dataset,
#         attempt.dataset,
#         classes_whitelist=[oc.name for oc in attempt_project_meta.obj_classes],
#         tags_whitelist=[tm.name for tm in attempt_project_meta.tag_metas],
#         obj_tags_whitelist=[tm.name for tm in attempt_project_meta.tag_metas],
#         iou_threshold=iou_threshold,
#     )

#     save_report(report, attempt)
#     update_report_status(report, attempt)
#     g.is_refreshing_report = False

#     return report


layout = Container(widgets=[Text("<h2>Exams</h2>"), exams_table.content])
