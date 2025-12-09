import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # app/
ROOT_DIR = os.path.dirname(BASE_DIR)                         # My_project/
STATIC_DIR = os.path.join(ROOT_DIR, "app", "static")
MODEL_DIR = os.path.join(ROOT_DIR, "app", "models")
