#!/usr/bin/python
# -*- coding:utf-8 -*-
import streamlit
import pandas
import streamlit.web.cli as stcli
import os, sys
import app


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())
