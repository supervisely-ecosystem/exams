import copy
import dateutil.parser
from supervisely.app.widgets import Text, Container, Button, Empty, Input, Select
from supervisely.api.labeling_job_api import LabelingJobApi
from supervisely import ProjectMeta

import src.globals as g
import src.utils as utils
from src.ui.expandable_table import ExpandableTable
from src.ui.report import calculate_report


class ExamsTable:

    def __init__(self, expandable_table: ExpandableTable, header: Container):
        self.table = expandable_table
        self._header = header
        self._exams = []
        self._filters = {}
    
    def update(self):
        self.table.loading = True
        self._header.loading = True
        exams_info = g.load_all_exams()
        self._exams = [
            ExpandableTable.Exam(
                name=exam_info["workspace_info"].name[7:-1],
                workspace=exam_info["workspace_info"],
                benchmark_project=exam_info["benchmark_project"],
                benchmark_project_meta=exam_info["benchmark_project_meta"],
                passmark=exam_info["benchmark_project"].custom_data["passmark"],
                created_at=exam_info["workspace_info"].created_at,
                created_by=g.api.user.get_info_by_id(exam_info["benchmark_project"].custom_data["created_by"]),
                attempts=exam_info["benchmark_project"].custom_data["attempts"],
                classes=exam_info["exam_project_meta"].obj_classes,
                tags=exam_info["exam_project_meta"].tag_metas,
                users = [ExpandableTable.ExamUser(
                    user_name=exam_project.custom_data["user_name"],
                    user_id=exam_project.custom_data["user_id"],
                    exam_project=exam_project,
                    labeling_jobs=utils.get_labeling_jobs(exam_project.id)
                ) for exam_project in exam_info["exam_projects"]],
            ) for exam_info in exams_info.values()
        ]
        self.table.set(self._exams)
        all_users = []
        creators = []
        for exam in self._exams:
            for user in exam._users:
                if user._user_id not in [u[0] for u in all_users]:
                    all_users.append((user._user_id, user._user_name))
            if exam._created_by.id not in [c[0] for c in creators]:
                creators.append((exam._created_by.id, exam._created_by.name))

        filter_by_assignee.set(items=[Select.Item(user[0], user[1]) for user in all_users])
        filter_by_assignee.set_value([])
        filter_by_creator.set(items=[Select.Item(creator[0], creator[1]) for creator in creators])
        filter_by_creator.set_value([])

        self.table.loading = False
        self._header.loading = False

    def search_by_name(self, text: str):
        self._header.loading = True
        self.table.loading = True
        exams = [exam for exam in self._exams if exam.get_name().find(text) != -1]
        self.table.set(exams)
        self.table.loading = False
        self._header.loading = False

    def filter(self):
        self._header.loading = True
        self.table.loading = True
        exams = self._exams.copy()
        for filter_name, filter_val in self._filters.items():
            if len(filter_val) == 0:
                continue
            if filter_name == "assignee":
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
                exams = filtered_exams
            if filter_name == "creator":
                filtered_exams = []
                for exam in exams:
                    if exam._created_by.id in filter_val:
                        filtered_exams.append(exam)
                exams = filtered_exams
            if filter_name == "status":
                filtered_exams = []
                for exam in exams:
                    users = []
                    for user in exam._users:
                        last_lj = user._labeling_jobs[0]
                        status = last_lj.status
                        exam_score = user._exam_project.custom_data.get("overall_score", None)
                        last_lj_id = user._exam_project.custom_data.get("last_labeling_job_id", None)
                        if exam_score is not None and last_lj_id == last_lj.id:
                            if exam_score*100 > exam._passmark:
                                status = f"passed"
                            else:
                                status = f"failed"
                    
                        if status in filter_val:
                            users.append(user)
                    if len(users) != 0:
                        new_exam = copy.deepcopy(exam)
                        new_exam._users = users
                        filtered_exams.append(new_exam)
                exams = filtered_exams    
        self.table.set(exams)
        self._header.loading = False
        self.table.loading = False

    def sort(self, val):
        self._header.loading = True
        self.table.loading = True

        key = lambda exam: dateutil.parser.isoparse(exam._created_at.rstrip("Z"))
        self.table.set(sorted(self.table._exams, key=key, reverse=val=="oldest"))
        
        self._header.loading = False
        self.table.loading = False

    def filter_changed(self, key, val):
        if type(val) != list:
            return
        self._filters[key] = val
        self.filter()

refresh_btn = Button(text="Refresh table", icon="zmdi zmdi-refresh", button_size="mini")
new_exam_button = Button(text="New Exam", icon="zmdi zmdi-plus", button_size="mini")
search_exam = Input(placeholder="Search by title", size="mini")
search_btn = Button("", button_size="mini", icon="zmdi zmdi-search", icon_gap=0)
filter_by_assignee = Select(placeholder="Assignee", items=[], multiple=True, size="mini")
filter_by_creator = Select(placeholder="Creator", items=[], multiple=True, size="mini")
filter_by_status = Select(
    items=[
        Select.Item(str(LabelingJobApi.Status.PENDING), "PENDING"),
        Select.Item(str(LabelingJobApi.Status.IN_PROGRESS), "IN PROGRESS"),
        Select.Item(str(LabelingJobApi.Status.ON_REVIEW), "ON REVIEW"),
        Select.Item(str(LabelingJobApi.Status.COMPLETED), "COMPLETED"),
        Select.Item(str(LabelingJobApi.Status.STOPPED), "STOPPED"),
        Select.Item("passed", "PASSED"),
        Select.Item("failed", "FAILED")
    ],
    placeholder="Status",
    multiple=True,
    size="mini"
)
select_sort = Select(items=[Select.Item("newest"), Select.Item("oldest")], placeholder="Sort", size="mini")
table_header = Container(widgets=[
    refresh_btn,
    filter_by_assignee,
    filter_by_creator,
    filter_by_status,
    select_sort,
    Empty(),
    Container(widgets=[search_exam, search_btn], direction="horizontal", fractions=[7,1], gap=5),
    new_exam_button
], fractions=[2, 2, 2, 2, 2, 7, 3, 2], direction="horizontal", overflow="wrap")
exams_table = ExamsTable(ExpandableTable([]), table_header)
table_and_header = Container(widgets=[table_header, exams_table.table])


def update_exams_table():
    exams_table.update()

@exams_table.table.refresh_clicked
def refresh_report(value_dict):
    g.is_refreshing_report = True

    workspace_id = value_dict["report"]["workspace_id"]
    project_id = value_dict["report"]["project_id"]

    benchmark_dataset = g.exams[workspace_id]["benchmark_dataset"]
    exam_dataset = utils.get_dataset_by_project_id(project_id)
    iou_threshold = g.exams[workspace_id]["benchmark_project"].custom_data["threshold"]
    pred_project_meta = ProjectMeta.from_json(g.api.project.get_meta(project_id))

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

@filter_by_assignee.value_changed
def filter_by_assignee_func(val):
    exams_table.filter_changed("assignee", val)

@filter_by_creator.value_changed
def filter_by_creator_func(val):
    exams_table.filter_changed("creator", val)

@filter_by_status.value_changed
def filter_by_status_func(val):
    exams_table.filter_changed("status", val)

@select_sort.value_changed
def sort_exams(val):
    exams_table.sort(val)

@search_btn.click
def filter_table_by_name():
    val = search_exam.get_value()
    exams_table.search_by_name(val)

@refresh_btn.click
def refresh_exams_table():
    update_exams_table()

layout = Container(widgets=[
    Text("<h2>Exams</h2>"),
    table_and_header
])
