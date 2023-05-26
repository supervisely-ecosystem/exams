import copy
import traceback
from typing import List, Optional

from supervisely.app.widgets import Widget, Empty, Text
from supervisely.app import DataJson, StateJson
from supervisely.sly_logger import logger


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

    def __init__(
        self,
        widget_id: Optional[str] = None,
    ):
        self._parsed_data = {}
        self._refresh_click_handled = False
        self._view_click_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)
    
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

    
    def insert_row(self, data, index=-1):
        table_data = self._parsed_data["data"]
        index = len(table_data) if index > len(table_data) or index < 0 else index

        self._parsed_data["data"].insert(index, data)
        DataJson()[self.widget_id]["table_data"] = self._parsed_data
        DataJson().send_changes()

    def pop_row(self, index=-1):
        index = (
            len(self._parsed_data["data"]) - 1
            if index > len(self._parsed_data["data"]) or index < 0
            else index
        )

        if len(self._parsed_data["data"]) != 0:
            popped_row = self._parsed_data["data"].pop(index)
            DataJson()[self.widget_id]["table_data"] = self._parsed_data
            DataJson().send_changes()
            return popped_row
