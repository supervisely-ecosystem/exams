from __future__ import annotations
import copy
import traceback
from typing import List, Optional, Union

from supervisely.app.widgets import Widget
from supervisely.app import DataJson, StateJson
from supervisely.sly_logger import logger
from supervisely import ObjClass, TagMeta, ObjClassCollection, TagMetaCollection, ProjectInfo, ProjectMeta, WorkspaceInfo
from supervisely.imaging.color import rgb2hex
from supervisely.api.labeling_job_api import LabelingJobApi, LabelingJobInfo
from supervisely.api.user_api import UserInfo
LabelingJobStatus = LabelingJobApi.Status

class ExpandableTable(Widget):
    class columns:
        EXAMS_TABLE_COLUMNS = [
            "exam",
            "passmark",
            "attempts",
            "classes",
            "tags",
            "created_at",
            "assignees",
            "benchmark_project",
        ]
        EXAM_USERS_TABLE_COLUMNS = [
            "user",
            "try",
            "started",
            "status",
            "report"
        ]

    class Routes:
        VIEW_CLICK = "view_clicked_cb"
        REFRESH_CLICK = "refresh_clicked_cb"

    class Exam:
        def __init__(
            self,
            name: str,
            workspace: WorkspaceInfo,
            benchmark_project: ProjectInfo,
            benchmark_project_meta: ProjectMeta,
            passmark: int,
            created_at: str,
            created_by: UserInfo,
            attempts: Optional[int] = None,
            classes: Union[List[ObjClass], ObjClassCollection] = [],
            tags: Union[List[TagMeta], TagMetaCollection] = [],
            users: List[ExpandableTable.ExamUser] = [],
        ) -> None:
            self._name = name
            self._workspace = workspace
            self._benchmark_project = benchmark_project
            self._benchmark_project_meta = benchmark_project_meta
            self._passmark = passmark
            self._created_at = created_at
            self._created_by = created_by
            self._attempts = attempts
            self._classes = classes
            self._tags = tags
            self._users = users
        
        def get_name(self):
            return self._name

    class ExamUser:
        def __init__(
            self, 
            user_name: str,
            user_id: int,
            exam_project: ProjectInfo,
            labeling_jobs: List[LabelingJobInfo] = [],
            
        ) -> None:
            self._user_name = user_name
            self._user_id = user_id
            self._exam_project = exam_project
            self._labeling_jobs = sorted(labeling_jobs, key=lambda lj: lj.id, reverse=True)

    def __init__(
        self,
        exams: List[ExpandableTable.Exam],
        widget_id: Optional[str] = None,
    ):
        self._exams = exams
        self.parse_exams()
        self._refresh_click_handled = False
        self._view_click_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)
    
    def parse_exams(self):
        def parse_exam_user_row(exam: ExpandableTable.Exam, user: ExpandableTable.ExamUser):
            STATUS_TEXT = {
                str(LabelingJobStatus.PENDING): 'PENDING',
                str(LabelingJobStatus.IN_PROGRESS): 'IN PROGRESS',
                str(LabelingJobStatus.ON_REVIEW): 'ON REVIEW',
                str(LabelingJobStatus.COMPLETED): 'COMPLETED',
                str(LabelingJobStatus.STOPPED): 'STOPPED',
            }
            last_lj = user._labeling_jobs[0]
            started_at = last_lj.started_at
            status = STATUS_TEXT[last_lj.status]

            exam_score = user._exam_project.custom_data.get("overall_score", None)
            last_lj_id = user._exam_project.custom_data.get("last_labeling_job_id", None)
            if exam_score is not None and last_lj_id == last_lj.id:
                if exam_score*100 > exam._passmark:
                    status = f"PASSED ({round(exam_score*100, 2)}%)"
                else:
                    status = f"FAILED ({round(exam_score*100, 2)}%)"
            
            return {
                "user_id": user._user_id,
                "user": user._user_name,
                "try": len(user._labeling_jobs),
                "attempts": "∞" if exam._attempts is None else exam._attempts,
                "started": started_at,
                "status": status,
                "report": {"workspace_id": exam._workspace.id, "project_id": user._exam_project.id},
                "loading": False,
                "passmark": exam._passmark,
            }

        def parse_exam_row(exam: ExpandableTable.Exam):
            return {
                "workspace_id": exam._workspace.id,
                "exam": exam._name,
                "passmark": exam._passmark,
                "attempts": "∞" if exam._attempts is None else exam._attempts,
                "classes": {
                    "gt": [{"class_name": oc.name, "color": rgb2hex(oc.color)} for oc in exam._benchmark_project_meta.obj_classes], 
                    "pred": [{"class_name": oc.name, "color": rgb2hex(oc.color)} for oc in exam._classes]},
                "tags": {
                    "gt": [{"tag_name": tm.name, "color": rgb2hex(tm.color)} for tm in exam._benchmark_project_meta.tag_metas],
                    "pred": [{"tag_name": tm.name, "color": rgb2hex(tm.color)} for tm in exam._tags],
                },
                "created_at": exam._created_at,
                "assignees": [user._user_name for user in exam._users],
                "benchmark_project": {
                    "name": exam._benchmark_project.name,
                    "url": exam._benchmark_project.url,
                    "preview_url": exam._benchmark_project.image_preview_url,
                    "description": f"{exam._benchmark_project.items_count} {exam._benchmark_project.type} in project"
                },
                "created_by": exam._created_by.name,
                "expandable_content": {
                    "table_data": {
                        "columns": ExpandableTable.columns.EXAM_USERS_TABLE_COLUMNS,
                        "data": [parse_exam_user_row(exam, user) for user in exam._users]
                    }
                }
            }

        self._parsed_data = {
            "columns": ExpandableTable.columns.EXAMS_TABLE_COLUMNS,
            "data": [parse_exam_row(exam) for exam in self._exams]
        }
    
    def get_json_data(self):
        return {"table_data": self._parsed_data, "loading": False}

    def get_json_state(self):
        return {
            "selected_row": {},
        }
    
    def clear_selection(self):
        StateJson()[self.widget_id]["selected_row"] = {}
        StateJson().send_changes()

    def _update_table_data(self, input_data):
        if input_data is not None:
            self._parsed_data = copy.deepcopy(input_data)
        else:
            self._parsed_data = {"columns": [], "data": [], "summaryRow": None}
            self._data_type = dict
    
    def read_json(self, value: dict) -> None:
        self._update_table_data(input_data=value)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()
        self.clear_selection()
    
    def set(self, exams: List[ExpandableTable.Exam]):
        self._exams = exams
        self.parse_exams()
        self.update_data()
        self.clear_selection()

    def get_selected_cell(self):
        return StateJson()[self.widget_id]["selected_row"]

    def view_clicked(self, func):
        route_path = self.get_route_path(ExpandableTable.Routes.VIEW_CLICK)
        server = self._sly_app.get_server()

        self._view_click_handled = True

        @server.post(route_path)
        async def _click():
            try:
                value_dict: dict = self.get_selected_cell()
                if value_dict is None:
                    return
                self.patch_expandable_row(value_dict, {**value_dict, "loading": True})
                func(value_dict)
                self.patch_expandable_row({**value_dict, "loading": True}, {**value_dict, "loading": False})
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click
    
    def refresh_clicked(self, func):
        route_path = self.get_route_path(ExpandableTable.Routes.REFRESH_CLICK)
        server = self._sly_app.get_server()

        self._refresh_click_handled = True
        
        @server.post(route_path)
        async def _click():
            try:
                value_dict = self.get_selected_cell()
                if value_dict is None:
                    return
                self.patch_expandable_row(value_dict, {**value_dict, "loading": True})
                func(value_dict)
                self.patch_expandable_row({**value_dict, "loading": True}, {**value_dict, "loading": False})
            except Exception as e:
                logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                raise e

        return _click


    def patch_expandable_row(self, old_row, new_row):
        for i, row in enumerate(DataJson()[self.widget_id]["table_data"]["data"]):
            for j, exp_row in enumerate(row["expandable_content"]["table_data"]["data"]):
                if exp_row == old_row:
                    DataJson()[self.widget_id]["table_data"]["data"][i]["expandable_content"]["table_data"]["data"][j] = new_row
                    DataJson().send_changes()
                    return
                
    @property
    def loading(self):
        return DataJson()[self.widget_id]["loading"]

    @loading.setter
    def loading(self, value: bool):
        DataJson()[self.widget_id]["loading"] = value
        DataJson().send_changes()
