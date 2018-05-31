#!/bin/bash

find . -name '*.py' | PYTHONPATH="${PYTHONPATH}:./src" entr make test
