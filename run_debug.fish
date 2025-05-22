#!/usr/bin/env fish
set -x FLASK_APP ai_service
set -x FLASK_DEBUG 1
flask run --host 0.0.0.0 --port 5000
