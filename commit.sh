#!/bin/env bash
source .venv/bin/activate
uv build
git add -f dist/*.whl

