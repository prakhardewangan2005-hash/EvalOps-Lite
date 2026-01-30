from src.api.app import app

# Railway/Gunicorn expects an "app" object.
# Keeping this file avoids changing Dockerfile/CMD.
