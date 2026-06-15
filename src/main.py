import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import supervisely as sly
import uvicorn
from src.ui import layout, load_dashboard
import src.globals as g

app = sly.Application(layout=layout, static_dir=g.TEMP_DATA_PATH)
load_dashboard()


if __name__ == "__main__":
	host = os.getenv("HOST", "127.0.0.1")
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run(app, host=host, port=port)
