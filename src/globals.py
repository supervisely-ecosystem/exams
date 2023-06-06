from __future__ import annotations
from typing import List
import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
server_address = sly.env.server_address()
api_token = sly.env.api_token()
team_id = sly.env.team_id()
user_id = sly.env.user_id()

api = sly.Api()

users = api.user.get_team_members(team_id)
is_refreshing_report = False
exams = {}


def is_benchmark_project(project: sly.ProjectInfo):
    return project.name.endswith(". Benchmark project")


def is_exam_project(project: sly.ProjectInfo, workspace_name):
    return project.name.startswith(
        workspace_name + ". User: "
    ) and not project.name.endswith("_DIFF")


def is_diff_project(project: sly.ProjectInfo):
    return project.name.endswith("_DIFF")


class Exam:
    class ExamUser:
        class Attempt:
            def __init__(
                self,
                project: sly.ProjectInfo,
                dataset: sly.DatasetInfo,
                labeling_job: sly.ProjectInfo,
                diff_project: sly.ProjectInfo,
                project_meta: sly.ProjectMeta,
            ):
                self.project = project
                self.dataset = dataset
                self.project_meta = project_meta
                self.labeling_job = labeling_job
                self.diff_project = diff_project

        def __init__(self, user_id, attempts: List[Exam.ExamUser.Attempt]):
            self.user_id = user_id
            self.attempts = attempts

    def __init__(
        self,
        workspace: sly.WorkspaceInfo,
        benchmark_project: sly.ProjectInfo,
        benchmark_dataset: sly.DatasetInfo,
        users: List[Exam.ExamUser],
    ):
        self.workspace = workspace
        self.benchmark_project = benchmark_project
        self.benchmark_dataset = benchmark_dataset
        self.benchmark_project_meta = sly.ProjectMeta.from_json(
            api.project.get_meta(benchmark_project.id)
        )
        self.users = {user.user_id: user for user in users}

    @classmethod
    def load_from_workspace(cls, workspace: sly.WorkspaceInfo):
        benchmark_project = None
        projects = [
            api.project.get_info_by_id(pr.id)
            for pr in api.project.get_list(workspace.id)
        ]
        user_attempt_projects = {}
        attempt_diff = {}
        for project in projects:
            if is_benchmark_project(project):
                benchmark_project = project
                continue
            if is_exam_project(project, workspace.name):
                if project.custom_data["user_id"] not in user_attempt_projects:
                    user_attempt_projects[project.custom_data["user_id"]] = []
                user_attempt_projects[project.custom_data["user_id"]].append(project)
                continue
            if is_diff_project(project):
                attempt_diff[project.custom_data["exam_project_id"]] = project
                continue
        exam_project_meta = sly.ProjectMeta.from_json(
            api.project.get_meta(list(user_attempt_projects.values())[0][0].id)
        )

        return Exam(
            workspace=workspace,
            benchmark_project=benchmark_project,
            benchmark_dataset=api.dataset.get_info_by_id(
                benchmark_project.custom_data["benchmark_dataset_id"]
            ),
            users=[
                cls.ExamUser(
                    user_id=user_id,
                    attempts=[
                        cls.ExamUser.Attempt(
                            project=project,
                            dataset=api.dataset.get_list(project.id)[0],
                            project_meta=exam_project_meta,
                            labeling_job=api.labeling_job.get_list(
                                team_id, project_id=project.id, show_disabled=True
                            )[0],
                            diff_project=attempt_diff[project.id]
                            if project.id in attempt_diff
                            else None,
                        )
                        for project in sorted(
                            projects, key=lambda x: x.id, reverse=True
                        )
                    ],
                )
                for user_id, projects in user_attempt_projects.items()
            ],
        )


@sly.timeit
def load_all_exams():
    global exams
    exam_workspaces = [
        ws for ws in api.workspace.get_list(team_id) if ws.name.startswith('Exam: "')
    ]
    for workspace in exam_workspaces:
        try:
            exam = Exam.load_from_workspace(workspace)
            exams[workspace.id] = exam
        except:
            pass
    return exams
