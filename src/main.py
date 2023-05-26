import supervisely as sly
from src.ui import layout, load_dashboard

app = sly.Application(layout = layout)
load_dashboard()
