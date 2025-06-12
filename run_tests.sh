#!/bin/bash

# Install test dependencies
pip install -r tests/requirements-test.txt

# Run tests with coverage
python -m pytest tests/ \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:coverage_report \
    -v
