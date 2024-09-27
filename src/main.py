import supervisely as sly
from src.ui import layout, load_dashboard
import src.globals as g

app = sly.Application(layout=layout, static_dir=g.TEMP_DATA_PATH)
load_dashboard()
