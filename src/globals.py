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

users = {user.id: user for user in api.user.get_team_members(team_id)}
is_refreshing_report = False
exams = {}

ALLOWED_MODALITIES = (sly.ProjectType.IMAGES, sly.ProjectType.VIDEOS)
TEMP_DATA_PATH = "/tmp/consensus-videos"

sly.fs.mkdir(TEMP_DATA_PATH)
