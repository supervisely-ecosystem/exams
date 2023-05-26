import os
from dotenv import load_dotenv
import supervisely as sly

import src.utils as utils

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


server_address = sly.env.server_address()
api_token = sly.env.api_token()
team_id = sly.env.team_id()
user_id = sly.env.user_id()

api = sly.Api()

gt_anns = None
pred_anns = None
diff_anns = None
gt_imgs = None
diff_imgs = None
pred_imgs = None
workspaces = []
users = api.user.get_team_members(team_id)
guides = []
is_refreshing_report = False

exams = {}


@sly.timeit
def load_all_exams():
    global exams
    exam_workspaces = utils.get_exams_workspaces()
    exam = {}
    for workspace in exam_workspaces:
        try:
            exam = {
                "benchmark_project": None,
                "benchmark_dataset": None,
                "benchmark_project_meta": None,
                "exam_projects": [],
                "exam_datasets": [],
                "exam_project_meta": None,
                "labeling_jobs": {},
            }
            # load project infos for benchmark project and exam projects
            projects = [api.project.get_info_by_id(pr.id) for pr in api.project.get_list(workspace.id)]
            for project in projects:
                if project.name == workspace.name + ". Benchmark project":
                    exam["benchmark_project"] = project
                    exam["benchmark_project_meta"] = sly.ProjectMeta.from_json(api.project.get_meta(project.id))
                elif project.name.startswith(workspace.name + ". User:") and not project.name.endswith("_DIFF"):
                    exam["exam_projects"].append(project)
                    if exam["exam_project_meta"] is None:
                        exam["exam_project_meta"] = sly.ProjectMeta.from_json(api.project.get_meta(project.id))

            # load dataset infos for benchmark project and exam projects
            for project in exam["exam_projects"]:
                dataset = api.dataset.get_list(project.id)[0]
                exam["exam_datasets"].append(dataset)
            exam["benchmark_dataset"] = api.dataset.get_info_by_name(exam["benchmark_project"].id, exam["exam_datasets"][0].name)
            
            # load labeling jobs for exam projects
            for project in exam["exam_projects"]:
                labeling_jobs = api.labeling_job.get_list(team_id, project_id=project.id)
                exam["labeling_jobs"][project.id] = labeling_jobs

            exams[workspace.id] = {
                "workspace_info": workspace,
                **exam
            }
        except:
            pass

    return exams
