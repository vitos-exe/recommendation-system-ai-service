#!/usr/bin/env fish
set -x FLASK_APP app
set -x FLASK_DEBUG 1
flask run --host 0.0.0.0 --port 5000
