from __future__ import annotations
from typing import List, Optional

import supervisely as sly

from src.constants import DEFAULT_EXAM_PASSMARK, DEFAULT_MATCHING_THRESHOLD


class Exam:
    class ExamUser:
        class Attempt:
            def __init__(
                self,
                project: sly.ProjectInfo,
                labeling_job: sly.ProjectInfo,
                dataset: sly.DatasetInfo,
                diff_project: sly.ProjectInfo,
                project_meta: sly.ProjectMeta,
            ):
                self.project = project
                self.project_meta = project_meta
                self.dataset = dataset
                self.labeling_job = labeling_job
                self.diff_project = diff_project

            def number(self):
                return get_attempt_number(self.project)

        def __init__(self, user_id, attempts: List[Exam.ExamUser.Attempt]):
            self.user_id = user_id
            self.attempts = sorted(attempts, key=lambda x: x.project.id, reverse=True)
        
        def get_last_attempt(self) -> Exam.ExamUser.Attempt:
            return self.attempts[0]

    def __init__(
        self,
        workspace: sly.WorkspaceInfo,
        benchmark_project: sly.ProjectInfo,
        benchmark_project_meta: sly.ProjectMeta,
        benchmark_dataset: sly.DatasetInfo,
        users: List[Exam.ExamUser]
    ):
        self.workspace = workspace
        self.benchmark_project = benchmark_project
        self.benchmark_dataset = benchmark_dataset
        self.benchmark_project_meta = benchmark_project_meta
        self._users = {user.user_id: user for user in users}
        self.attempt_project_meta = self._get_attempt_project_meta()

    @classmethod
    def load_from_workspace(cls, workspace: sly.WorkspaceInfo, api: Optional[sly.Api]):
        if api is None:
            api = sly.Api()
        benchmark_project = None
        workspace_projects = [
            api.project.get_info_by_id(pr.id)  # to load all fields of ProjectInfo
            for pr in api.project.get_list(workspace.id)
        ]
        user_attempt_projects = {}
        attempt_diff = {}
        for project in workspace_projects:
            if is_benchmark_project(project):
                benchmark_project = project
                continue
            if is_attempt_project(project, workspace.name):
                user_attempt_projects.setdefault(
                    get_attempt_project_user_id(project, api), []
                ).append(project)
                continue
            if is_diff_project(project):
                attempt_diff[get_attempt_project_id_of_diff_project(project, api)] = project
                continue

        if benchmark_project is None:
            raise RuntimeError("Can't find benchmark project in workspace.")
        if len(user_attempt_projects) == 0:
            raise RuntimeError("Can't find exam projects in workspace.")

        benchmark_project_meta = get_project_meta(benchmark_project.id, api)

        return Exam(
            workspace=workspace,
            benchmark_project=benchmark_project,
            benchmark_project_meta=benchmark_project_meta,
            benchmark_dataset=api.dataset.get_info_by_id(
                get_benchmark_dataset_id(benchmark_project)
            ),
            users=[
                cls.ExamUser(
                    user_id=user_id,
                    attempts=[
                        cls.ExamUser.Attempt(
                            project=project,
                            project_meta=get_project_meta(project.id, api),
                            dataset=api.dataset.get_info_by_id(get_exam_dataset_id(project, api)),
                            labeling_job=api.labeling_job.get_list(
                                workspace.team_id, project_id=project.id, show_disabled=True
                            )[0],
                            diff_project=attempt_diff[project.id]
                            if project.id in attempt_diff
                            else None,
                        )
                        for project in projects
                    ],
                )
                for user_id, projects in user_attempt_projects.items()
            ],
        )
    
    def name(self):
        return self.workspace.name[7:-1]

    def get_user(self, user_id):
        return self._users[user_id]

    def get_all_users(self):
        return list(self._users.values())
        
    def get_passmark(self):
        try:
            return self.benchmark_project.custom_data["passmark"]
        except KeyError:
            sly.logger.warning(f"Can't find passmark for exam: {self.name()} (workspace id: {self.workspace.id}). Using default value {DEFAULT_EXAM_PASSMARK}.")
            return DEFAULT_EXAM_PASSMARK

    def created_at(self):
        return self.workspace.created_at

    def created_by(self):
        try:
            return self.benchmark_project.custom_data["created_by"]
        except KeyError:
            sly.logger.warning(f"Can't find creator for exam: {self.name()} (workspace id: {self.workspace.id}).")
            return None

    def max_attempts(self):
        try:
            return self.benchmark_project.custom_data["attempts"]
        except KeyError:
            sly.logger.warning(f"Can't find max attempts for exam: {self.name()} (workspace id: {self.workspace.id}).")
            return None
        
    def iou_threshold(self):
        try:
            return self.benchmark_project.custom_data["threshold"]
        except KeyError:
            sly.logger.warning(f"Can't find iou threshold for exam: {self.name()} (workspace id: {self.workspace.id}).")
            return DEFAULT_MATCHING_THRESHOLD
    
    def reviewer_id(self):
        try:
            return self.benchmark_project.custom_data["reviewer_id"]
        except KeyError:
            sly.logger.warning(f"Can't find reviewer for exam: {self.name()} (workspace id: {self.workspace.id}).")
            return None
        
    def _get_attempt_project_meta(self):
        try:
            return list(self._users.values())[0].attempts[0].project_meta
        except:
            raise RuntimeError("Can't load project meta for exam project.")


def is_benchmark_project(project: sly.ProjectInfo):
    try:
        return project.custom_data["is_benchmark_project"]
    except KeyError:
        return project.name.endswith(". Benchmark project")


def is_attempt_project(project: sly.ProjectInfo, workspace_name):
    try:
        return project.custom_data["is_attempt_project"]
    except KeyError:
        return project.name.startswith(
            workspace_name + ". User: "
        ) and not project.name.endswith("_DIFF")


def is_diff_project(project: sly.ProjectInfo):
    try:
        return project.custom_data["is_diff_project"]
    except KeyError:
        return project.name.endswith("_DIFF")
    

def parse_user_login_from_project_name(project_name: str):
    return project_name.split("User: ")[1].split(".")[0]


def get_attempt_project_user_id(project: sly.ProjectInfo, api: sly.Api):
    try:
        return project.custom_data["user_id"]
    except KeyError:
        user_login = parse_user_login_from_project_name(project.name)
        return api.user.get_info_by_login(user_login).id


def get_attempt_project_id_of_diff_project(project: sly.ProjectInfo, api: sly.Api):
    try: 
        project_id = project.custom_data.get("exam_project_id", None)
        if project_id is not None:
            return project_id
        else:
            return project.custom_data["attempt_project_id"]
    except KeyError:
        attempt_project_name = project.name.split("_DIFF")[0]
        attempt_project = api.project.get_info_by_name(project.workspace_id, attempt_project_name)
        if attempt_project is None:
            raise RuntimeError(f"Attempt project not found for diff project {project.name} (id: {project.id}).")
        return attempt_project.id


def get_project_meta(project_id: int, api: sly.Api):
    return sly.ProjectMeta.from_json(api.project.get_meta(project_id))


def get_exam_dataset_id(project: sly.ProjectInfo, api: sly.Api):
    try:
        return project.custom_data["exam_dataset_id"]
    except KeyError:
        return api.dataset.get_list(project.id)[0].id


def get_benchmark_dataset_id(project: sly.ProjectInfo):
    try:
        return project.custom_data["benchmark_dataset_id"]
    except KeyError:
        raise RuntimeError(f"Can't find benchmark dataset id for benchmark project: {project.name} (id: {project.id})")


def get_attempt_number(project: sly.ProjectInfo):
    try:
        return project.custom_data["attempt"]
    except KeyError:
        try:
            return int(project.name.split("Attempt: ")[1])
        except:
            raise RuntimeError(f"Can't find attempt number for project: {project.name} (id: {project.id})")
