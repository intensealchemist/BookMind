#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install -r requirements.prod.txt --no-cache-dir

# Download spaCy language model
python -m spacy download en_core_web_sm

python manage.py collectstatic --no-input
python manage.py migrate
