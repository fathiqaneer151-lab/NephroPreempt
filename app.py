"""
NephroPreempt Streamlit application.

Single-file clinical decision-support web page that loads the bundled retrained
models and age/sex reference curve from the app folder or Desktop/software.
"""


from __future__ import annotations


import html

import logging
import os


import json


import math


import re


import textwrap


import zipfile


from dataclasses import dataclass


from datetime import date, datetime, time


from pathlib import Path, PureWindowsPath


from typing import Any


import numpy as np


import pandas as pd


import streamlit as st


try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional UI dependency
    go = None


try:
    from scipy.signal import savgol_filter
except Exception as exc:  # pragma: no cover - fallback is supplied
    savgol_filter = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


APP_TITLE = "NephroPreempt"


APP_SUBTITLE = "Predictive Clinical Analytics for Early Kidney Risk"


PROJECT_LOGO_DATA_URI = "data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiID8+PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2aWV3Qm94PSIwIDEzNSAxMDIyIDY0MCI+PHBhdGggZmlsbD0iIzA2NUU1NiIgZD0iTTI5NC4zOTkgMzY4LjczMUMyOTguNzQyIDMxMi41NTYgMzI0Ljk3NiAyNjAuMzM1IDM2Ny40NTEgMjIzLjMxN0M0MTUuMzk2IDE4Mi4zMDIgNDcwLjYwNSAxNjQuMDg4IDUzMy40NTQgMTY5LjI2QzU5My4xIDE3NC4xNjggNjQ2LjgwNiAyMDEuMDY2IDY4NS40NjQgMjQ2Ljk5MkM3MjMuNjc2IDI5Mi45MzYgNzQyLjMxNSAzNTIuMDQ5IDczNy4zNjggNDExLjYwMUM3MzIuNjQ2IDQ3MC45OTEgNzA0LjEyNyA1MjUuOTUzIDY1OC4yODYgNTY0LjAwN0M2MTMuNTQyIDYwMS40MDYgNTUzLjgxIDYxOC41MjkgNDk2LjA5OSA2MTIuOTY2QzQzNS4zNTQgNjA3LjExMSAzODMuMTE4IDU3Ny45MDUgMzQ0LjQzOCA1MzEuMjg3QzMxOS44MjYgNTAxLjgzNSAzMDMuMzQgNDY2LjQ2IDI5Ni42MTcgNDI4LjY3MkMyOTUuODc3IDQyNC4zMDkgMjk0LjE1OCA0MTIuMjU1IDI5NC4zODQgNDA4LjQ0NUMyOTMuMDQ1IDQwMC4wMTIgMjkzLjM3IDM3Ny4xIDI5NC4zOTkgMzY4LjczMVpNMjk2LjEwMyA0MDAuMzI1QzMwMS4xNzYgNTE5LjY1NiA0MDAuNTM2IDYxMy4xNDEgNTE5Ljk1NCA2MTAuOTQyQzYzOS4zNzMgNjA4Ljc0MyA3MzUuMjIzIDUxMS42NjIgNzM1Ljg5OCAzOTIuMjI2QzczNi41NzQgMjcyLjc4OSA2NDEuODI5IDE3NC42MyA1MjIuNDQzIDE3MS4wNzlDNDAzLjA1NyAxNjcuNTI5IDMwMi42NDYgMjU5Ljg4NCAyOTYuMjIzIDM3OS4xNUMyOTUuODQzIDM4Ni4yMDIgMjk1LjgwMyAzOTMuMjY5IDI5Ni4xMDMgNDAwLjMyNVoiLz48cGF0aCBmaWxsPSIjMDY1RTU2IiBkPSJNNTExLjY1OCAxODAuMjMyQzYyNy44MjQgMTc3LjgzOSA3MjMuOTI5IDI3MC4wODUgNzI2LjI5NiAzODYuMjUyQzcyOC42NjMgNTAyLjQxOSA2MzYuMzk3IDU5OC41MDMgNTIwLjIyOSA2MDAuODQ1QzQwNC4wOTggNjAzLjE4NiAzMDguMDQ5IDUxMC45NTQgMzA1LjY4MyAzOTQuODIzQzMwMy4zMTYgMjc4LjY5MiAzOTUuNTI3IDE4Mi42MjQgNTExLjY1OCAxODAuMjMyWk01MzAuNjc3IDU5Mi40OEM2NDIuMTM4IDU4NC4zNzUgNzI1LjkzOSA0ODcuNDcgNzE3Ljg3OCAzNzYuMDA2QzcwOS44MTggMjY0LjU0MSA2MTIuOTQ2IDE4MC43MDIgNTAxLjQ3OSAxODguNzE3QzM4OS45NDggMTk2LjczOCAzMDYuMDUgMjkzLjY3NiAzMTQuMTE1IDQwNS4yMDRDMzIyLjE4IDUxNi43MzIgNDE5LjE1MiA2MDAuNTkgNTMwLjY3NyA1OTIuNDhaIi8+PHBhdGggZmlsbD0iIzA2NUU1NiIgZD0iTTU2NS43MzQgMjcyLjA1MUM1NzAuNzM3IDI3MS44MjkgNTc0LjU4MyAyNzEuNzQ4IDU3OS41IDI3Mi45NDNDNjA0LjUyNCAyNzkuMDI3IDYyOS4yMTggMzA2LjA0IDY0Mi4yMjUgMzI3LjE5OEM2NjEuNyAzNTkuMjg4IDY2Ny41OTMgMzk3LjgxMiA2NTguNjAyIDQzNC4yNTZDNjUyLjE0MSA0NjEuMzExIDYzNS4wODkgNDkxLjE0OCA2MTAuODgzIDUwNS45NTVDNjA3Ljg5IDUwNy43NDEgNjA0LjY4IDUwOS4xMzYgNjAxLjMzMSA1MTAuMTA1QzU5MC42ODQgNTEzLjEzNCA1NzguMzQyIDUxMi40NjEgNTY4LjUzNSA1MDYuOTMxQzU0NS45ODEgNDk0LjIxMyA1NTIuNTggNDYwLjA2NSA1NTkuMzA5IDQ0MC4wMDNDNTY0LjY3MyA0MjQuMTg5IDU3Ni44OSA0MDcuMjY0IDU3NC40NzcgMzg5LjkyM0M1NzEuNTYzIDM3NC4zNjMgNTU5LjYzMSAzNjIuODUgNTUxLjk3NCAzNDguODU1QzU0MC42MzMgMzI4LjEyOCA1MzIuMzM1IDI5Ny44MTUgNTUxLjQ5MyAyNzguODYxQzU1NS40MTQgMjc0Ljk4MyA1NjAuNDQ0IDI3My4zNjEgNTY1LjczNCAyNzIuMDUxWiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik00NTEuODY2IDI3Mi4wNjJDNDY3LjQ2IDI3MC45NDIgNDc5LjcyMiAyNzguNzgxIDQ4NC44NzcgMjkzLjU5OUM0OTEuODEyIDMxMy41MzQgNDg0LjIyMyAzMzQuNzEgNDc0LjAzOCAzNTIuMDFDNDY2LjkxOSAzNjQuMTAzIDQ1NS45NjQgMzc1LjM3OCA0NTMuNjg3IDM4OS42NTRDNDUwLjk3NCA0MDguMDc3IDQ2My43MTcgNDI1LjUyIDQ2OS4xNDkgNDQyLjM2N0M0NzUuNzUxIDQ2Mi44NDIgNDgxLjgyNiA0OTQuMzIzIDQ1OC44OSA1MDYuOTk0QzQ0Ny43MzYgNTEzLjE1NiA0MzcuMjM3IDUxMi43MTkgNDI1LjI4MyA1MDkuNTY4QzM5NS41IDUwMC4zNzUgMzczLjc5MSA0NTQuNzEzIDM2Ny45NzQgNDI1Ljk2M0MzNjAuNTc1IDM4OS42OTcgMzY4LjEwMSAzNTEuOTc5IDM4OC44NTIgMzIxLjMyOUM0MDIuOSAzMDAuMjk3IDQyNi40ODEgMjc3LjA3MyA0NTEuODY2IDI3Mi4wNjJaIi8+PHBhdGggZmlsbD0iIzA2NUU1NiIgZD0iTTUxMC41MTMgMjA1LjIzN0M1NDAuNTYxIDIwMi45NjkgNTU0LjgzNCAyMzQuNDczIDU0Mi43NiAyNTkuMTIyQzUzNy42MTEgMjY5LjYzNSA1MjkuMjE0IDI3OC43MTMgNTI0LjcxNiAyODkuNDU4QzUxNC41ODUgMzEzLjYyNCA1MjIuOTU2IDM0MS41MyA1MzcuMTc3IDM2Mi4xMzZDNTQ1LjUxMyAzNzQuMTMyIDU1Ny4xMiAzODUuNDAyIDU1My4zMzQgNDAxLjQ1OUM1NTAuMzQzIDQxNC4xNDYgNTQyLjgwOSA0MjYuMjAzIDUzOC4yNjIgNDM4LjQ4QzUzMy45MzQgNDUwLjE2OCA1MzEuMDM2IDQ2MS4yMTUgNTMxLjE5MSA0NzMuNzcxQzUzMS4zNjYgNDgzLjY1NCA1MzMuNDcyIDQ5My40MDcgNTM3LjM5MSA1MDIuNDgxQzU0MS4wOTMgNTExLjExNSA1NDUuODI1IDUxOS42NDQgNTQ3LjEwMSA1MjkuMDQ5QzU0OS44OTUgNTQ5LjY1NSA1NDAuNjcgNTczLjQ5OCA1MTguMDM2IDU3Ni45OTRDNDg2LjYyOCA1NzcuNzEyIDQ3NC4zMDQgNTQ2LjYzNyA0ODIuNzA4IDUxOS41MjJDNDg1LjQ1MyA1MTAuNjY0IDQ5MC45NDMgNTAzLjI3OCA0OTMuNTI0IDQ5My43MzdDNTAwLjUyOSA0NjcuODUgNDkyLjk5NCA0NDUuMDgxIDQ4MS4yMTkgNDIyLjA5NkM0NzcuOTkzIDQxNS43NzkgNDc1LjU1MiA0MDkuMDc0IDQ3My45NzYgNDAyLjA1NUM0NzAuNTY1IDM4Ni44NTkgNDgwLjQ0NyAzNzcuMzE4IDQ4OC4zMjkgMzY1LjgzNUM1MDYuMjU3IDMzOS42NjcgNTE1LjczIDMwNy4zMTggNDk2LjkyNCAyNzguOTQ1QzQ5MC44NTYgMjY5Ljc4OSA0ODIuMzQ1IDI1OC4yNjcgNDgwLjQxNCAyNDcuNDhDNDc4LjY2OSAyMzcuODg5IDQ4MC44MDggMjI3Ljk5OCA0ODYuMzYxIDIxOS45ODZDNDkyLjYwOCAyMTEuMDM0IDQ5OS45OSAyMDcuMDYgNTEwLjUxMyAyMDUuMjM3WiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik0yOTQuMzk5IDM2OC43MzFDMjk0LjU3IDM3MS42NTggMjk0LjE0MyAzNzguMjg0IDI5NS4yNjIgMzgwLjIxQzI5NS40NzIgMzc5Ljk4OSAyOTYuMDg3IDM3OS4zNzcgMjk2LjIyMyAzNzkuMTVDMjk1Ljg0MyAzODYuMjAyIDI5NS44MDMgMzkzLjI2OSAyOTYuMTAzIDQwMC4zMjVMMjk1LjExMiAzOTkuNjY4QzI5My43OTMgNDAxLjU1IDI5NC40ODYgNDA1Ljg2MiAyOTQuMzg0IDQwOC40NDVDMjkzLjA0NSA0MDAuMDEyIDI5My4zNyAzNzcuMSAyOTQuMzk5IDM2OC43MzFaIi8+PHBhdGggZmlsbD0iIzA2NUU1NiIgZD0iTTQxNC4yMjggNjcyLjg3M0M0MjUuNjg3IDY3MS41NTUgNDM5Ljg5OSA2NzQuNiA0NDkuNDI0IDY4MS42MzNDNDYzLjkwNCA2OTIuMzI0IDQ2Ni44MTMgNzE0LjkxOCA0NTUuNTY5IDcyOS4wNTJDNDQ3LjQ4NCA3MzkuMjE3IDQzNy43MjcgNzQyLjEyOCA0MjUuMTM4IDc0My42MDhDNDAwLjE5NiA3NDQuNjExIDM3Ni4yNjIgNzM2Ljc5MSAzNzUuODYyIDcwNy4yOTlDMzc1LjU1NSA2ODQuNjggMzk0LjEzMiA2NzQuOTkgNDE0LjIyOCA2NzIuODczWk00MjQuMTYgNzI5Ljk2OUM0MjcuMyA3MjkuMzE3IDQyOS42MjggNzI4Ljg5MSA0MzIuNjIyIDcyNy41NDFDNDQ3LjUxOSA3MjAuODI3IDQ0OS4wMzIgNjk4Ljg4NCA0MzQuNzU5IDY5MC41MzZDNDI4Ljc2OCA2ODcuMDMyIDQyMS43NDMgNjg2LjI2NyA0MTQuNzg5IDY4Ni44NzVDNDAxLjQ1IDY4OC42MjYgMzkyLjIgNjk3LjI1MiAzOTQuMzE5IDcxMS41MjNDMzk1LjE2MiA3MTcuMzUyIDM5OC40MTMgNzIyLjU2IDQwMy4yNzkgNzI1Ljg3OUM0MDkuNTg1IDczMC4yNDIgNDE2LjgxOSA3MzAuODY1IDQyNC4xNiA3MjkuOTY5WiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik03NzEuNDAyIDY3My44NThDNzgxLjE0OSA2NzIuOTQyIDc4OC4yNjcgNjc0LjgyNCA3OTYuMjIgNjgwLjQxOUM4MDQuOTgyIDY3NC4xNTUgODEwLjA5MSA2NzMuOTM2IDgyMC4zMDEgNjc0LjEwOUM4NDAuNDMgNjc0LjQ1IDg0Mi45NzggNjg4LjY0MiA4NDIuNDYgNzA1LjI2NUM4NDIuMTQ2IDcxNS4zNTQgODQyLjgyIDcyNi4xOTQgODQyLjUyNyA3MzYuMzY4Qzg0Mi40NzcgNzM4LjA5NiA4NDEuNDc0IDc0MS4zMDkgODM5Ljc5NSA3NDIuMzA5QzgzNS40ODQgNzQ0LjI4NiA4MjcuOTU0IDc0NC4zOTMgODI2LjA2NiA3MzguODg4QzgyMi45NjggNzI5LjM2NiA4MjguMzgxIDY5OC44NTEgODIyLjgxMiA2OTEuNTI3QzgyMS40NzQgNjg5Ljc2OCA4MTkuNDYxIDY4OC42NDggODE3LjI2MSA2ODguNDM4QzgwNS4wMDggNjg3LjM4IDgwNS41NDUgNjk4LjAyNiA4MDUuNTcxIDcwNy4wMDNDODA1LjU4NyA3MTIuNDc3IDgwNS42NCA3MTcuOTg0IDgwNS42OSA3MjMuNDU5QzgwNS41ODYgNzI4LjI3OSA4MDcuMDA3IDczNS4yNSA4MDQuODY5IDczOS42MzRDODA0LjEwOSA3NDEuMTkzIDgwMi44NjkgNzQyLjI0MyA4MDEuMjI4IDc0Mi43ODNDNzk4LjY1MiA3NDMuNjMxIDc5NC4wMDUgNzQzLjQ4MyA3OTEuNjM3IDc0Mi4xMDdDNzg0Ljc3NSA3MzguMTE3IDc5MC4zIDcwNS4wMDIgNzg3LjM5IDY5NS4yNDdDNzg2LjcxOCA2OTIuOTkyIDc4NS41NTMgNjkwLjk2IDc4My4zOTUgNjg5Ljg2N0M3ODAuNjc3IDY4OC40OSA3NzYuNTE0IDY4Ny4zODYgNzczLjUyMyA2ODguNDNDNzcwLjQ5IDY4OS40OSA3NjguNzM0IDY5Mi4zMDYgNzY3Ljg1MSA2OTUuMjQ5Qzc2NS42MTggNzAyLjY5MSA3NjYuNzk2IDcyMS45MDQgNzY2Ljc3NyA3MzAuNTkyQzc2Ni43NyA3MzMuNTYzIDc2Ny4yNDIgNzM4LjM2NiA3NjUuNjMyIDc0MC45NDlDNzY0Ljg5NyA3NDIuMTI3IDc2My41ODYgNzQyLjY3NCA3NjIuMjc4IDc0Mi45M0M3NTkuNTQ0IDc0My40NjUgNzU0LjQyOCA3NDMuNTk4IDc1Mi4xNDUgNzQxLjc1MkM3NTAuODg1IDc0MC43MzMgNzUwLjQ4NSA3MzkuMjc0IDc1MC4yMzEgNzM3Ljc0OEM3NDguODg1IDcyOS42NzggNzQ4Ljk5NyA3MDAuMzUgNzUwLjE3MSA2OTIuMDI1Qzc1MC42MzUgNjg4LjczNCA3NTEuNDgxIDY4NC41MDIgNzUzLjUyMiA2ODEuODQ2Qzc1Ny44NTcgNjc2LjIwMyA3NjQuODA3IDY3NC44OTkgNzcxLjQwMiA2NzMuODU4WiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik01NDYuMjkzIDY3NS4yN0M1NTcuMDQyIDY3NC40NTcgNTg1LjAxNyA2NzMuOTMgNTkzLjg3NyA2NzcuODgxQzYxMi45MjIgNjg2LjM3NSA2MTIuMzk3IDcwOC42ODIgNTkxLjQ2NSA3MTcuMDQ5QzU5NS4yNDYgNzIwLjMzOCA1OTguOTI3IDcyNS45NDggNjAyLjEzNSA3MjkuOTE5QzYwNC41NTkgNzMyLjkyMiA2MDguNTMgNzM3LjczMSA2MDYuNzI0IDc0MS42MDVDNjAzLjkxMSA3NDMuNTg0IDU5OS43OTYgNzQzLjA2NSA1OTYuMjggNzQzLjAzMkM1OTQuNDk1IDc0Mi45NDQgNTkyLjUwNSA3NDIuNDYgNTkwLjczMyA3NDIuMDk5QzU4Ny4zNTEgNzM4LjI0NSA1NzguNjc1IDcyNC44ODIgNTc1Ljg0NCA3MjAuMjIzQzU3MC4zMzggNzIwLjExOSA1NjUuMzM3IDcyMC4yOCA1NTkuODQ4IDcyMC40NzZDNTU5Ljg2NSA3MjQuOTExIDU2MC45OTUgNzM4LjM2MSA1NTguMDY0IDc0MS42NjdDNTU1LjkxMiA3NDQuMDk0IDU0Ni41MDEgNzQzLjY2NiA1NDMuOTA4IDc0MC40ODFDNTQyLjAwNyA3MzIuMTc3IDU0My40ODcgNjg4Ljk4OCA1NDMuNDAyIDY3OC4yMjFDNTQzLjM5MiA2NzYuOTgzIDU0NS4yMDUgNjc2LjAyOSA1NDYuMjkzIDY3NS4yN1pNNTU5LjgxIDcwOC40OTlDNTY3LjM0OSA3MDguNTI4IDU3OS45NDEgNzA5LjU1NSA1ODYuMTQgNzA1LjIzQzU4OC40ODEgNzAyLjUyMyA1ODkuMzk3IDcwMC45MDIgNTg5LjIzNyA2OTcuMTk4QzU4OC43MTkgNjg1LjIxOCA1NjguNDYzIDY4OC4wNTkgNTU5LjczMiA2ODguMTY0QzU1OS44OTggNjk0Ljk0MSA1NTkuOTI0IDcwMS43MjEgNTU5LjgxIDcwOC40OTlaIi8+PHBhdGggZmlsbD0iIzA2NUU1NiIgZD0iTTMwNS44MiA2NzUuMjYzQzMxOC44NTUgNjc0LjY0MiAzMzUuMzc3IDY3NC4yNTggMzQ4LjExMSA2NzYuNTQzQzM2MS41ODEgNjc4Ljk1OCAzNjguODgxIDY5MS40NjUgMzY1LjAxNCA3MDQuNDM1QzM2Mi45NyA3MTEuMjkxIDM1Ni42ODQgNzE0LjMwMiAzNTAuNzQxIDcxNy4xMjRDMzU0LjUyNiA3MjEuMTI0IDM1Ny43MTkgNzI1Ljk5MiAzNjEuMTY0IDczMC4yOTVDMzcyLjAyOCA3NDMuODY0IDM2MS45MTggNzQzLjc1NyAzNTEuMDIyIDc0Mi41NjVDMzQ2LjcxNCA3NDAuMDY1IDMzOC4xNTcgNzI1LjE1NyAzMzUuMTU2IDcyMC4yMTlDMzMwLjAyMSA3MjAuMTYxIDMyNC44ODUgNzIwLjE4OSAzMTkuNzUxIDcyMC4zMDJDMzE5LjcxNiA3MjguNDA2IDMyMy4zNzMgNzQyLjcxOSAzMTIuOTA2IDc0My4xOThDMzAxLjUxNyA3NDMuNzE5IDMwMy4xNjggNzM4LjE1NiAzMDMuMTM3IDcyOS41MDZDMzAzLjA3OCA3MTIuOTMzIDMwMy4wNDEgNjk2LjY3OSAzMDMuMjE1IDY4MC4xNDFDMzAzLjI0NyA2NzcuMzQ4IDMwMy44NjcgNjc3LjAzIDMwNS44MiA2NzUuMjYzWk0zMTkuODcgNzA4LjQyNUMzMjcuNTUgNzA4LjU1OCAzMzguMzQyIDcwOS41NTkgMzQ0LjcxNiA3MDUuMDQzQzM0Ny4wOTEgNzAyLjEwNSAzNDguMDcxIDcwMC41OTEgMzQ3Ljc3OCA2OTYuNjg5QzM0Ni45MTIgNjg1LjE5NiAzMjguMzcgNjg4LjE4OCAzMTkuNzg2IDY4OC4wODVDMzE5LjgxOSA2OTQuNzc0IDMxOS45NjIgNzAxLjc2MiAzMTkuODcgNzA4LjQyNVoiLz48cGF0aCBmaWxsPSIjMDY1RTU2IiBkPSJNODU3LjU0MSA2NzUuMjM3Qzg2NS4zMSA2NzQuNzA3IDg3My41NDggNjc1LjAyNCA4ODEuMzY1IDY3NC45NkM4OTEuMTk2IDY3NC44NzkgOTAzLjA0NyA2NzQuNzg2IDkxMS4yNjIgNjgwLjkzM0M5MjAuNTI5IDY4Ny44NjcgOTIxLjI4NiA3MDMuODE3IDkxMy44NDQgNzEyLjM2M0M5MDguMjgxIDcxOC43NTIgOTAyLjI5IDcyMC4xOTYgODk0LjM3NSA3MjAuOTMzQzg4Ny4xNjMgNzIxLjIzNCA4NzkuMDk1IDcyMS4wNjggODcxLjgxNSA3MjEuMDYxQzg3MS44MDggNzI5LjI3MSA4NzUuMjA1IDc0NC40NTUgODYzLjY4MSA3NDMuMjIxQzg1OC42NTUgNzQyLjY4MyA4NTQuNzcyIDc0Mi44MyA4NTQuNTQxIDczNi40NDdDODU0LjAwMiA3MjAuMjg1IDg1NC4zNDMgNzA0LjA0NiA4NTQuMjU1IDY4Ny44NjhDODU0LjIyNyA2ODIuNzQyIDg1My4xNjQgNjc4LjM3NCA4NTcuNTQxIDY3NS4yMzdaTTg3MS43ODYgNzA5LjEzNkM4NzguNTAzIDcwOS4xMDggODkyLjQ4IDcwOS44ODkgODk3Ljg5NyA3MDUuOTE5QzkwMC43MDIgNzAyLjU5NSA5MDEuNDMzIDcwMS4zMzYgOTAxLjE3NCA2OTYuODU1QzkwMC41MDIgNjg1LjI1MiA4ODAuMTYzIDY4OC4wMDQgODcxLjc2IDY4OC4wOTlMODcxLjc4NiA3MDkuMTM2WiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik00NzYgNjc1LjIzOEM0ODEuMDIgNjc0LjgyOCA0ODguNDk0IDY3NC44OSA0OTMuNzAyIDY3NC45NThDNTA1LjExNCA2NzUuMjExIDUxOC42MDggNjczLjYwNCA1MjggNjgxLjQ4NUM1MzYuOTI3IDY4OC45NzYgNTM3LjcxIDcwNC41OTQgNTI5LjY0OCA3MTMuMDI5QzUyMy4wMyA3MTkuOTUyIDUxNS4zNzggNzIwLjg3NyA1MDYuMzc3IDcyMS4wNjFDNTAwLjg1MyA3MjEuMTIyIDQ5NS4xNDYgNzIxLjAzMSA0ODkuNjA3IDcyMS4wMDhDNDg5LjgyNiA3MjUuNzc1IDQ5MS4yOTYgNzM3LjQ4IDQ4OC4zNzMgNzQxLjIzOUM0ODcuMTU2IDc0Mi44MDQgNDg1Ljk1MSA3NDMuMDMgNDg0LjA3MyA3NDMuMTM1QzQ4MS40OTQgNzQzLjI4IDQ3NS40NzMgNzQzLjE4MSA0NzMuNzE0IDc0MC44MTFDNDcxLjcwMyA3MzguMTA0IDQ3MS42NDggNjg0LjM1NSA0NzIuNjQxIDY3OC45MDFDNDcyLjk4NyA2NzYuOTk2IDQ3NC41MTIgNjc2LjIzIDQ3NiA2NzUuMjM4Wk00ODkuNjE2IDcwOS4xMUM0OTYuNzUzIDcwOS4xMjMgNTA4LjMzMyA3MDkuOTU5IDUxNC4yNTggNzA1Ljg4M0M1MTYuOTYxIDcwMi43OTggNTE3LjQ3MiA3MDEuODE1IDUxNy40ODIgNjk3LjY1MkM1MTcuNTA5IDY4NS4zMDQgNDk4LjE3MSA2ODcuOTk1IDQ4OS42MTMgNjg4LjA4OEM0ODkuNjE5IDY5NS4wNjkgNDg5LjY4NiA3MDIuMTM4IDQ4OS42MTYgNzA5LjExWiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik0xNjUuODQ4IDY3NS4zNDlDMTg0LjAxOCA2NzQuMjAzIDIyMi4zMzUgNjcwLjc5OSAyMjMuNzQ0IDY5NS4zODNDMjI0Ljg1NyA3MTQuNzk1IDIxMi43NTIgNzE5Ljk3NSAxOTYuNDkxIDcyMC43MTZDMTkxLjQ2MSA3MjEuMTExIDE4NC45MDggNzIxLjAwNyAxNzkuNzYgNzIxLjA2NkMxNzkuODc4IDczMC40OTggMTgzLjEgNzQ1LjYxMSAxNjkuNDMgNzQzLjE1OEMxNjYuNzIyIDc0Mi42NzIgMTY0Ljg5MiA3NDIuNjEyIDE2My4xODQgNzQwLjI1OEMxNjAuOTk3IDcyOS44MTIgMTYyLjU0NyA2OTEuOTY0IDE2Mi40NSA2NzkuMDk2QzE2Mi40MzcgNjc3LjM3OSAxNjQuNDAzIDY3Ni4xODYgMTY1Ljg0OCA2NzUuMzQ5Wk0xNzkuNTcyIDcwOC41NzhDMTg3LjQwNCA3MDguNTI5IDE5Ny41NTcgNzA5LjYwNSAyMDMuNzM3IDcwNS4yNjRDMjA0LjU2MSA3MDQuMDc3IDIwNS41NzkgNzAyLjY4MyAyMDYuMzQyIDcwMS40ODVDMjA2LjM2NiA2OTcuNzk1IDIwNi42NCA2OTQuMjMzIDIwMy42MjIgNjkxLjQ5OEMxOTguNTIxIDY4Ni44NzMgMTg2Ljc5IDY4OS4yNTIgMTc5LjU0MiA2ODguNjkyQzE3OS43NDkgNjk0Ljc5NiAxNzkuODc2IDcwMi40NTQgMTc5LjU3MiA3MDguNTc4WiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik02NTYuOTgzIDY3NC45NDJDNjYxLjMzNSA2NzQuODc1IDY3MC43MjcgNjc0LjAyNSA2NzMuOTQyIDY3Ni40NDVDNjc3LjU4NiA2NzkuMTg2IDY3Ni45OTYgNjg4LjM1OSA2NzAuODM5IDY4OC43NDdDNjU5LjIzNiA2ODkuNDc3IDY0Ny4zODMgNjg5LjA4IDYzNS43MjggNjg4Ljk5N0M2MzUuNzY4IDY5Mi43NzUgNjM1LjkzMSA2OTYuNTg1IDYzNS4zNDMgNzAwLjMxMUM2NDAuNTM3IDcwMC4wODYgNjY1Ljg3MiA2OTguODQ2IDY2OC4yNTcgNzAyLjQyNUM2NzguODg0IDcxOC4zNjggNjQ0LjIyMyA3MTQuMTA5IDYzNS44NzYgNzEzLjcxMUw2MzUuODM4IDcyOS4xMTdDNjQyLjA5MiA3MjguOTY5IDY0OC42MDEgNzI5LjAyMyA2NTQuODc2IDcyOC45OTFDNjYwLjI5NSA3MjkuMDMgNjcwLjYyMiA3MjcuMzk1IDY3NC45NzYgNzMwLjg1M0M2NzYuMTAxIDczMS43NDcgNjc2Ljc4MSA3MzIuOTQyIDY3Ni45MzggNzM0LjM3OUM2NzcuMTY2IDczNi40NjUgNjc2LjExNCA3MzkuNjUgNjc0LjY3MiA3NDEuMTc1QzY3Mi44ODggNzQzLjA2MiA2NjguMzA5IDc0Mi45NjYgNjY1LjgwOCA3NDMuMDM5QzY1Ny4zNzggNzQzLjI4OSA2MjkuNzEgNzQzLjgyOSA2MjIuOTQgNzQyLjQyNUM2MjEuOTA5IDc0Mi4yMTEgNjIwLjk4OCA3NDEuNzYyIDYyMC4yNSA3NDAuOTkzQzYxNy41MzcgNzM4LjE2NCA2MTcuOTEgNzMxLjM3MiA2MTcuODQxIDcyNy42ODlDNjE3LjY4NCA3MTkuMjI1IDYxNy45MzMgNzEwLjc0NCA2MTcuODk4IDcwMi4yNzdDNjE3Ljg3MiA2OTUuOTI2IDYxNy40MjQgNjg5LjMxNiA2MTcuOTk0IDY4My4wMDZDNjE4LjE0NyA2ODEuMzEzIDYxOC41MTUgNjc3LjcwMyA2MTkuODkzIDY3Ni41NjJDNjIzLjQ0NSA2NzMuNjIxIDY1MC42MjIgNjc0Ljk2NiA2NTYuOTgzIDY3NC45NDJaIi8+PHBhdGggZmlsbD0iIzA2NUU1NiIgZD0iTTY4OC4xMDcgNjc1LjI3NEM2OTAuNjYgNjc1LjAzNyA2OTguMjYyIDY3NC43OTIgNzAwLjgxMiA2NzUuMDA4QzcwNi41MTMgNjc1LjQ5IDczOC4zNDcgNjczLjc1NCA3NDAuMzY0IDY3Ni40MTZDNzUwLjM3IDY4OS42MjMgNzMxLjkzNCA2ODkuMzg0IDcyMy41OTUgNjg5LjA5OEM3MTcuODg0IDY4OC45MDMgNzA4Ljc1IDY4OS4wMzkgNzAyLjUzMiA2ODkuMDAzQzcwMi41NTcgNjkzLjA5NiA3MDIuNTU0IDY5Ni4yMDEgNzAyLjIwMSA3MDAuMjZDNzA4LjMxOSA3MDAuMjQyIDcyOS45ODEgNjk5LjQ5MiA3MzQuNDE2IDcwMS4yMDNDNzM2LjkzNiA3MDMuNjA0IDczNy45MDMgNzA5Ljg2NyA3MzUuMjg1IDcxMS42MTNDNzI5LjAyNyA3MTUuNzg4IDcxMS4yNTYgNzE0LjA3MiA3MDMuOTk3IDcxNC4wNDdDNzAzLjYxOSA3MTQuMDIyIDcwMy4yNTEgNzE0LjE4NSA3MDIuODgyIDcxNC4yNzJDNzAyLjA3NCA3MTcuODQ2IDcwMi44ODggNzI0LjQ4MSA3MDIuMjIxIDcyOS4xOTZDNzE0LjAwOSA3MjguNzM4IDcyNi40OTkgNzI4LjU5NCA3MzguMjQgNzI5LjQ1NEM3NDUuMTc4IDcyOS45NjMgNzQyLjc3NCA3NDIuMTg1IDczOS41NzQgNzQyLjI2QzcyOS41MjYgNzQyLjQ5NSA2OTQuNjg0IDc0NS4wOTIgNjg2Ljc3MiA3NDEuMTYxQzY4NC42MzggNzM3Ljk3NSA2ODQuNzA3IDczMC4yMDggNjg0LjcxOSA3MjYuMjQ0QzY4NC43NjQgNzEwLjk1NCA2ODQuMzg3IDY5NS41NzkgNjg1LjAxNSA2ODAuMzA1QzY4NS4xMjggNjc3LjU2OSA2ODYuMTI2IDY3Ni44NTYgNjg4LjEwNyA2NzUuMjc0WiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik0yNzUuNzk4IDcwMS4yMzVDMjc1LjQ4MSA2OTUuNjM3IDI3My45NSA2ODMuMTk5IDI3Ni4yNDEgNjc4LjM5M0MyNzcuMDYxIDY3Ni42NzEgMjc4LjUwMSA2NzUuNTE4IDI4MC4zMDIgNjc0LjkyOEMyODIuOTQ4IDY3NC4wNjEgMjg4LjIxMSA2NzMuOTExIDI5MC42MjMgNjc1LjRDMjkyLjA4OCA2NzYuMzA0IDI5Mi43OCA2NzcuMjk4IDI5My4wNDkgNjc4Ljk2NUMyOTQuMDE0IDY4NC45NDUgMjk0LjEyNSA3MzguMDcyIDI5MS44MzkgNzQwLjk3NUMyOTAuMzM5IDc0Mi44OCAyODcuODc0IDc0Mi44ODIgMjg1LjY1MyA3NDMuMTMyQzI3NC40MDQgNzQzLjE1NCAyNzUuMzc4IDczNy4zMiAyNzUuMzcyIDcyNy44NzZDMjc1LjM2OSA3MjMuNTI2IDI3NS42NjEgNzE5LjEzIDI3NS45MzcgNzE0Ljc4NkMyNjYuODczIDcxNC44NjYgMjU3LjQwOCA3MTQuMzk5IDI0OC40NjIgNzE0Ljg4MUMyNDguNTA5IDcyMy4wNDYgMjQ5LjE2OSA3MzIuMjkyIDI0Ny44ODEgNzQwLjI0OUMyNDcuMDk2IDc0NS4wOTcgMjMyLjA1IDc0My43MjYgMjMxLjk0MiA3MzkuODRDMjMxLjcwNCA3MzEuMjE3IDIyOS4yNTEgNjgyLjk1NSAyMzEuOTAxIDY3Ni45MzlDMjMzLjQ1MSA2NzUuNDc3IDIzMy45MTQgNjc1LjQ3NSAyMzYuMDIgNjc0Ljc5NUwyMzYuNzkgNjc0Ljc0QzI1My4zNDYgNjczLjY5OSAyNDguNTE4IDY4OC40MTQgMjQ4LjUzNCA3MDEuMjI1TDI3NS43OTggNzAxLjIzNVoiLz48cGF0aCBmaWxsPSIjMDY1RTU2IiBkPSJNMTA0LjI3OCA2NzUuMjI5QzExMC43ODggNjc0LjkyNiAxNDguMjA4IDY3NC4xMSAxNTEuNjU0IDY3Ni4xMTNDMTUzLjI3NyA2NzcuMDU3IDE1My44NTEgNjc5LjUyNyAxNTMuOTgzIDY4MS4yNjJDMTU0LjE0IDY4My4zMiAxNTMuODUgNjg1LjY0NSAxNTIuMjkxIDY4Ny4xNTRDMTQ4LjY2NyA2OTAuNjY1IDEyMy4wMzQgNjg5LjA3NiAxMTcuMTY1IDY4OS4wNDlDMTE3LjMxIDY5Mi41NTcgMTE3LjAxIDY5Ni43NzkgMTE2Ljg0NiA3MDAuMzQyQzEyMS42MzMgNzAwLjE1MSAxNDYuNTY1IDY5OS42NjkgMTQ5LjEyNCA3MDEuMjU4QzE1MC40NTEgNzAyLjk0NyAxNTAuODE0IDcwMy45MjggMTUwLjkxNiA3MDYuMDQzQzE1MS4yNzIgNzEyLjM1NCAxNDcuODk3IDcxMy45OTQgMTQyLjEyOSA3MTQuMDU5QzEzNC4wOTkgNzE0LjE0OSAxMjYuMDI3IDcxMy45MzggMTE4LjAxMiA3MTMuODA2TDExNy4yNjEgNzE0LjI1M0MxMTcuMzA5IDcxOS4yMDggMTE3LjI4OCA3MjQuMTY0IDExNy4xOTkgNzI5LjExOUwxMzguNzQ1IDcyOC44N0MxNDIuMjM5IDcyOC44MzcgMTUwLjE4MSA3MjguMDM2IDE1Mi41ODEgNzMwLjk4MUMxNTMuODk0IDczMi41OTIgMTU0LjE0OSA3MzUuMTM2IDE1NC4wNzYgNzM3LjEzMkMxNTQuMDExIDczOC45MzcgMTUzLjM2NiA3NDAuNDkgMTUxLjkyNCA3NDEuNjRDMTQ4LjgxIDc0NC4xMjQgMTEwLjk3OCA3NDMuMzE5IDEwNC45NzcgNzQyLjMwNkMxMDQuMDkyIDc0Mi4xNTcgMTAxLjk1MSA3NDEuODQyIDEwMS40ODcgNzQxLjAxOEMxMDAuNDkyIDczOS4yNTEgMTAwLjI5MSA3MzYuMTA3IDEwMC4xNTcgNzM0LjA4OUM5OS41MDE2IDcyNC4xODYgMTAwLjEzMiA3MTMuOTE4IDEwMC4xMzkgNzAzLjk3N0MxMDAuMTQ0IDY5Ny4zNzQgOTkuMTgwNiA2ODMuMzA3IDEwMC41MjMgNjc3LjY4OUMxMDEuNzY1IDY3Ni4wNzcgMTAyLjMwNCA2NzYuMDQgMTA0LjI3OCA2NzUuMjI5WiIvPjxwYXRoIGZpbGw9IiMwNjVFNTYiIGQ9Ik01My40OTU5IDY3NC4zNjJDOTYuMzkwMyA2NzAuNDI5IDkwLjkwMDMgNzAwLjk2NCA5MC45OTI1IDczMC45OEM5MS4wMjM1IDc0MS4wNTUgOTAuMzMxMSA3NDUuNTYzIDc3Ljg5ODUgNzQyLjY4MkM3MS43MDUgNzQxLjI0NyA3NC43MzQ2IDcyMy40NDIgNzQuMDQ2OCA3MTguODFDNzMuMTA2NSA3MDkuMzEzIDc3LjQ5NiA2OTUuMzIxIDY2LjE5NzggNjg5Ljg2M0w2NS42Njc0IDY4OS42MTNDMzMuNzY4MyA2NzkuNjYzIDQ5LjYzNTYgNzI1LjkxOSA0My44Mjk2IDc0MC4yODRDNDEuNzI0MSA3NDUuNDk0IDI4LjY3ODIgNzQ1LjQxNyAyOC4yMTQzIDczNi45MDhDMjcuNDkwMSA3MzAuNTQ3IDI4LjA1MzMgNzI0LjA5MSAyNy44OTc2IDcxNy42ODlDMjcuMzcwMiA2OTYuMDAxIDI2LjI4MzQgNjc4LjI1NCA1My40OTU5IDY3NC4zNjJaIi8+PHBhdGggZmlsbD0iIzA2NUU1NiIgZD0iTTkyNi4zMDUgNjc0Ljg3M0M5MzMuMTUyIDY3NC42MjEgOTg0Ljc4MiA2NzMuODA5IDk4Ny40MDIgNjc1Ljk5OUM5ODguODkzIDY3Ny4yNDUgOTg5LjgwMSA2NzkuNzE5IDk5MC4wMDMgNjgxLjYwNEM5OTAuMTc5IDY4My4yNDcgOTg5Ljc3NCA2ODQuNzk2IDk4OC42MTIgNjg2LjAyNkM5ODQuMjI0IDY5MC42NzUgOTcwLjk5MiA2ODguOTE1IDk2NC43NzYgNjg4Ljk5NEw5NjQuODEzIDcyNC4xOTJDOTY0Ljg1NiA3MjguMDA2IDk2Ni4wMjUgNzM5LjAxMiA5NjMuMTA2IDc0MS43MTVDOTYyLjAyMyA3NDIuNzE3IDk1OS42MzYgNzQyLjk2NCA5NTguMjQ3IDc0My4xQzk1NS4zMjQgNzQzLjM4NyA5NTEuMjU4IDc0My4zMDYgOTQ4Ljk3NCA3NDEuMTQ0Qzk0NS43NzkgNzM4LjEyMSA5NDcuMTggNjk2LjY3OCA5NDcuMTYxIDY4OC45NzJDOTQwLjU1IDY4OC4yNDYgOTI0LjUwNCA2OTEuNzIgOTIxLjk1OCA2ODMuNzY1QzkyMC42NTkgNjc5LjcwNiA5MjEuODIyIDY3Ni4xMTMgOTI2LjMwNSA2NzQuODczWiIvPjwvc3ZnPg=="


REQUIRED_ASSET_FILENAMES = {
    "xgb": "xgboost_kidney_model.ubj",
    "lstm": "lstm_uacr_model.keras",
    "reference": "NephroPreempt_Reference_Curve.csv",
}


ASSET_FILENAME_OPTIONS = {
    "xgb": ("xgboost_kidney_model_retrained.ubj", REQUIRED_ASSET_FILENAMES["xgb"]),
    "lstm": ("lstm_uacr_model_retrained.keras", REQUIRED_ASSET_FILENAMES["lstm"]),
    "reference": ("NephroPreempt_Reference_Curve_18_90.csv", REQUIRED_ASSET_FILENAMES["reference"]),
}


def candidate_software_dirs() -> list[Path]:
    """Resolve beside the entry point first, on Windows or Community Cloud Linux."""
    app_dir = Path(__file__).resolve().parent
    return list(dict.fromkeys([app_dir, Path.cwd(), Path.cwd() / "software", app_dir / "software"]))


def directory_has_required_assets(directory: Path) -> bool:
    return all(
        any((directory / filename).exists() for filename in filenames)
        for filenames in ASSET_FILENAME_OPTIONS.values()
    )


def resolve_software_dir() -> Path:
    for directory in candidate_software_dirs():
        if directory_has_required_assets(directory):
            return directory
    return Path(__file__).resolve().parent


def resolve_asset_path(primary_filename: str, preferred_filenames: tuple[str, ...] = ()) -> Path:
    filenames = (*preferred_filenames, primary_filename)
    for filename in filenames:
        for directory in candidate_software_dirs():
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return resolve_software_dir() / primary_filename


def display_path(path: Path) -> str:
    return str(path)


def searched_asset_locations() -> str:
    return ", ".join(display_path(path) for path in candidate_software_dirs())


SOFTWARE_DIR = resolve_software_dir()


XGB_MODEL_PATH = resolve_asset_path(
    REQUIRED_ASSET_FILENAMES["xgb"],
    preferred_filenames=("xgboost_kidney_model_retrained.ubj",),
)


LSTM_MODEL_PATH = resolve_asset_path(
    REQUIRED_ASSET_FILENAMES["lstm"],
    preferred_filenames=("lstm_uacr_model_retrained.keras",),
)


REFERENCE_CURVE_PATH = resolve_asset_path(
    REQUIRED_ASSET_FILENAMES["reference"],
    preferred_filenames=("NephroPreempt_Reference_Curve_18_90.csv",),
)


DEFAULT_XGB_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "Baseline_Systolic_BP",
    "Diabetes",
    "Hypertension",
    "Smoker",
    "ACEi_ARB_Usage",
]


WEEKS_IN_SERIES = 12


AGE_MIN = 18.0


AGE_MAX = 90.0


BMI_MIN = 16.0


BMI_MAX = 55.0


UACR_MIN = 0.1


UACR_MAX = 5000.0


WEIGHT_MIN = 30.0


WEIGHT_MAX = 250.0


HEIGHT_MIN = 100.0


HEIGHT_MAX = 250.0


SBP_MIN = 85


SBP_MAX = 198


SAVGOL_WINDOW_LENGTH = 5


SAVGOL_POLY_ORDER = 2


MICROALBUMINURIA_MIN = 30.0


MACROALBUMINURIA_MIN = 300.0


MICRO_SEVERITY_FLOOR = 0.11


MACRO_SEVERITY_FLOOR = 0.28


LOW_RISK_MAX = 10.0


MODERATE_RISK_MAX = 25.0


HIGH_RISK_MAX = 50.0


META_INTERCEPT = 0.0


MODEL_XGB_AUC = 0.8602357679433451


MODEL_LSTM_AUC = 0.9247207491906133


MODEL_XGB_SKILL = MODEL_XGB_AUC - 0.50


MODEL_LSTM_SKILL = MODEL_LSTM_AUC - 0.50


MODEL_TREND_RELIABILITY_WEIGHT = MODEL_LSTM_SKILL / (MODEL_LSTM_SKILL + MODEL_XGB_SKILL)


DYNAMIC_TREND_WEIGHT_MIN = 0.28


DYNAMIC_TREND_WEIGHT_MAX = 0.82


LSTM_UACR_MIN = np.asarray(
    [0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)


LSTM_UACR_MAX = np.asarray(
    [
        233.662087,
        265.565932,
        302.901067,
        344.060171,
        1023.707280,
        1017.397679,
        1016.973755,
        600.291787,
        698.619874,
        805.402566,
        935.460433,
        1095.950153,
    ],
    dtype=np.float32,
)


RISK_BANDS = {
    "Low Risk": {
        "color": "#008f78",
        "dark_color": "#5eead4",
        "surface": "#e1faf4",
        "directive": (
            "Continue the established care plan and routine kidney-function and UACR "
            "monitoring, interpreted alongside the full clinical record."
        ),
    },
    "Moderate Risk": {
        "color": "#b7791f",
        "dark_color": "#fbbf24",
        "surface": "#fff7e6",
        "directive": (
            "Consider closer clinical and laboratory follow-up. Review modifiable risk "
            "factors, medications, and the longitudinal record before changing care."
        ),
    },
    "High Risk": {
        "color": "#c2410c",
        "dark_color": "#fb923c",
        "surface": "#fff1e8",
        "directive": (
            "Arrange prompt clinician review, verify the UACR trajectory, and apply local "
            "kidney-care pathways when deciding on repeat testing or treatment changes."
        ),
    },
    "Critical Risk": {
        "color": "#991b1b",
        "dark_color": "#f87171",
        "surface": "#fee2e2",
        "directive": (
            "Urgent clinician review is recommended. Assess the patient directly and "
            "follow local escalation or referral protocols; this score does not diagnose "
            "imminent kidney failure."
        ),
    },
}


@dataclass(frozen=True)
class StaticInputs:
    dob: date
    age: float
    gender: int
    height_cm: float
    weight_kg: float
    baseline_systolic_bp: int
    diabetes: int
    hypertension: int
    smoker: int
    acei_arb_usage: int

    @property
    def bmi(self) -> float:
        height_m = self.height_cm / 100.0
        return round(self.weight_kg / (height_m * height_m), 2)

    @property
    def sex_label(self) -> str:
        return "Male" if self.gender == 1 else "Female"


@dataclass(frozen=True)
class ReferenceProfile:
    age_used: float
    age_was_clamped: bool
    sex_label: str
    reference_value: float
    column_name: str
    row_age: float
    curve: np.ndarray


@dataclass(frozen=True)
class UACRProcessing:
    raw: np.ndarray
    smoothed: np.ndarray
    smoothing_delta: np.ndarray
    smoothing_delta_max: float
    z_scores: np.ndarray
    beta1: float
    raw_slope: float
    weights: np.ndarray
    uacr_opt: float
    mean: float
    std: float


@dataclass(frozen=True)
class PredictionResult:
    p_xgb: float
    p_trend: float
    p_trend_model_raw: float
    trend_probability_note: str
    fused_probability: float
    xgb_fusion_weight: float
    trend_fusion_weight: float
    uacr_relative_change: float
    fusion_note: str
    final_probability: float
    final_percent: float
    risk_level: str
    severity_floor_label: str
    severity_floor_applied: bool
    static_feature_order: list[str]
    lstm_tensor_shape: tuple[int, ...]
    reference: ReferenceProfile
    uacr_processing: UACRProcessing


@dataclass(frozen=True)
class AssetBundle:
    xgb_model: Any | None
    lstm_model: Any | None
    reference_curve: pd.DataFrame | None
    xgb_feature_names: list[str]
    lstm_input_shape: tuple[Any, ...] | None
    lstm_config: dict[str, Any]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def decimal_age_from_dob(dob: date) -> float:
    dob_dt = datetime.combine(dob, time.min)
    current_dt = datetime.now()
    age = (current_dt - dob_dt).total_seconds() / (365.25 * 86400.0)
    return round(float(age), 2)


def safe_logit(probability: float) -> float:
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def coerce_probability(raw_prediction: Any) -> float:
    arr = np.asarray(raw_prediction, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("The model returned an empty prediction.")
    if arr.ndim == 0:
        probability = float(arr)
    else:
        flattened = arr.reshape(arr.shape[0], -1)
        probability = float(flattened[0, 1] if flattened.shape[1] >= 2 else flattened[0, 0])
    if not np.isfinite(probability):
        raise ValueError("The model returned a non-finite probability.")
    if 1.0 < probability <= 100.0:
        probability /= 100.0
    return float(np.clip(probability, 0.0, 1.0))


def read_lstm_config(path: Path) -> dict[str, Any]:
    if not path.exists() or not zipfile.is_zipfile(path):
        return {}
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("config.json") as file:
                config = json.loads(file.read().decode("utf-8"))
            with zf.open("metadata.json") as file:
                metadata = json.loads(file.read().decode("utf-8"))
        input_shape = None
        layers = config.get("config", {}).get("layers", [])
        for layer in layers:
            if layer.get("class_name") == "InputLayer":
                input_shape = layer.get("config", {}).get("batch_shape")
                break
        return {
            "keras_version": metadata.get("keras_version"),
            "date_saved": metadata.get("date_saved"),
            "input_shape": input_shape,
            "model_name": config.get("config", {}).get("name"),
        }
    except Exception:
        return {}


def make_attention_custom_objects_with_tensorflow(tf: Any) -> dict[str, Any]:
    class UACRAttentionLayer(tf.keras.layers.Layer):
        def __init__(self, attention_units: int = 64, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.attention_units = int(attention_units)

        def build(self, input_shape: tuple[Any, ...]) -> None:
            feature_dim = int(input_shape[-1])
            self.attention_w = self.add_weight(
                name="attention_w",
                shape=(feature_dim, self.attention_units),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.attention_u = self.add_weight(
                name="attention_u",
                shape=(feature_dim, self.attention_units),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.attention_b = self.add_weight(
                name="attention_b",
                shape=(self.attention_units,),
                initializer="zeros",
                trainable=True,
            )
            self.context_u = self.add_weight(
                name="context_u",
                shape=(self.attention_units, 1),
                initializer="glorot_uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs: Any) -> Any:
            query = tf.expand_dims(tf.matmul(inputs[:, -1, :], self.attention_u), axis=1)
            score = tf.tanh(tf.tensordot(inputs, self.attention_w, axes=1) + query + self.attention_b)
            attention_logits = tf.tensordot(score, self.context_u, axes=1)
            attention_weights = tf.nn.softmax(attention_logits, axis=1)
            return tf.reduce_sum(inputs * attention_weights, axis=1)

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update({"attention_units": self.attention_units})
            return config

    return {
        "primary": {"UACRAttentionLayer": UACRAttentionLayer},
    }


def make_attention_custom_objects_with_keras(keras_module: Any) -> dict[str, Any]:
    ops = keras_module.ops

    class UACRAttentionLayer(keras_module.layers.Layer):
        def __init__(self, attention_units: int = 64, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.attention_units = int(attention_units)

        def build(self, input_shape: tuple[Any, ...]) -> None:
            feature_dim = int(input_shape[-1])
            self.attention_w = self.add_weight(
                name="attention_w",
                shape=(feature_dim, self.attention_units),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.attention_u = self.add_weight(
                name="attention_u",
                shape=(feature_dim, self.attention_units),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.attention_b = self.add_weight(
                name="attention_b",
                shape=(self.attention_units,),
                initializer="zeros",
                trainable=True,
            )
            self.context_u = self.add_weight(
                name="context_u",
                shape=(self.attention_units, 1),
                initializer="glorot_uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs: Any) -> Any:
            query = ops.expand_dims(ops.matmul(inputs[:, -1, :], self.attention_u), axis=1)
            score = ops.tanh(ops.matmul(inputs, self.attention_w) + query + self.attention_b)
            attention_logits = ops.matmul(score, self.context_u)
            attention_weights = ops.softmax(attention_logits, axis=1)
            return ops.sum(inputs * attention_weights, axis=1)

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update({"attention_units": self.attention_units})
            return config

    return {"primary": {"UACRAttentionLayer": UACRAttentionLayer}}


def load_lstm_model(path: Path) -> Any:
    errors: list[str] = []

    def try_load(loader: Any, custom_objects: dict[str, Any]) -> Any:
        try:
            return loader(str(path), compile=False, custom_objects=custom_objects, safe_mode=False)
        except TypeError:
            return loader(str(path), compile=False, custom_objects=custom_objects)

    try:
        import tensorflow as tf

        custom_sets = make_attention_custom_objects_with_tensorflow(tf)
        for custom_objects in custom_sets.values():
            try:
                return try_load(tf.keras.models.load_model, custom_objects)
            except Exception as exc:
                errors.append(str(exc))
    except Exception as exc:
        errors.append(f"TensorFlow unavailable: {exc}")

    try:
        import keras

        custom_sets = make_attention_custom_objects_with_keras(keras)
        for custom_objects in custom_sets.values():
            try:
                return try_load(keras.models.load_model, custom_objects)
            except Exception as exc:
                errors.append(str(exc))
    except Exception as exc:
        errors.append(f"Keras unavailable: {exc}")

    raise RuntimeError(f"Unable to load {path.name}. " + " | ".join(errors[-4:]))


def load_xgb_model(path: Path) -> Any:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def model_input_shape(model: Any) -> tuple[Any, ...] | None:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape:
        shape = shape[0]
    if shape is None:
        return None
    return tuple(shape)


@st.cache_resource(show_spinner=False)
def load_assets() -> AssetBundle:
    errors: list[str] = []
    warnings: list[str] = []
    xgb_model = None
    lstm_model = None
    reference_curve = None
    xgb_feature_names = DEFAULT_XGB_FEATURES.copy()
    lstm_input_shape = None
    lstm_config = read_lstm_config(LSTM_MODEL_PATH)
    searched_locations = searched_asset_locations()

    if not XGB_MODEL_PATH.exists():
        errors.append(
            f"Missing required model: {XGB_MODEL_PATH}. Searched: {searched_locations}"
        )
    else:
        try:
            xgb_model = load_xgb_model(XGB_MODEL_PATH)
            feature_names = getattr(xgb_model, "feature_names", None)
            if feature_names:
                xgb_feature_names = [str(name) for name in feature_names]
        except Exception as exc:
            errors.append(f"Could not load {XGB_MODEL_PATH.name}: {exc}")

    if not LSTM_MODEL_PATH.exists():
        errors.append(
            f"Missing required model: {LSTM_MODEL_PATH}. Searched: {searched_locations}"
        )
    else:
        try:
            lstm_model = load_lstm_model(LSTM_MODEL_PATH)
            lstm_input_shape = model_input_shape(lstm_model)
        except Exception as exc:
            errors.append(f"Could not load {LSTM_MODEL_PATH.name}: {exc}")

    if not REFERENCE_CURVE_PATH.exists():
        errors.append(
            f"Missing required reference CSV: {REFERENCE_CURVE_PATH}. Searched: {searched_locations}"
        )
    else:
        try:
            reference_curve = pd.read_csv(REFERENCE_CURVE_PATH)
            required = {
                "Age",
                "Male_Median_P50",
                "Female_Median_P50",
            }
            missing = sorted(required.difference(reference_curve.columns))
            if missing:
                errors.append("Reference CSV is missing columns: " + ", ".join(missing))
        except Exception as exc:
            errors.append(f"Could not load {REFERENCE_CURVE_PATH.name}: {exc}")

    if savgol_filter is None:
        warnings.append(
            "SciPy is unavailable; the app will use a safe identity fallback for the "
            f"window={SAVGOL_WINDOW_LENGTH}/polyorder={SAVGOL_POLY_ORDER} Savitzky-Golay step."
        )
    if go is None:
        warnings.append("Plotly is unavailable; charts will be rendered as a table.")

    return AssetBundle(
        xgb_model=xgb_model,
        lstm_model=lstm_model,
        reference_curve=reference_curve,
        xgb_feature_names=xgb_feature_names,
        lstm_input_shape=lstm_input_shape,
        lstm_config=lstm_config,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def static_value_aliases(inputs: StaticInputs) -> dict[str, float]:
    values = {
        "Age": float(inputs.age),
        "Gender": float(inputs.gender),
        "BMI": float(inputs.bmi),
        "Baseline_Systolic_BP": float(inputs.baseline_systolic_bp),
        "Diabetes": float(inputs.diabetes),
        "Hypertension": float(inputs.hypertension),
        "Smoker": float(inputs.smoker),
        "ACEi_ARB_Usage": float(inputs.acei_arb_usage),
    }
    aliases = {
        "age": values["Age"],
        "patient_age": values["Age"],
        "decimal_age": values["Age"],
        "gender": values["Gender"],
        "sex": values["Gender"],
        "bmi": values["BMI"],
        "body_mass_index": values["BMI"],
        "baseline_systolic_bp": values["Baseline_Systolic_BP"],
        "systolic_bp": values["Baseline_Systolic_BP"],
        "baseline_bp": values["Baseline_Systolic_BP"],
        "sbp": values["Baseline_Systolic_BP"],
        "diabetes": values["Diabetes"],
        "diabetic": values["Diabetes"],
        "hypertension": values["Hypertension"],
        "hypertensive": values["Hypertension"],
        "smoker": values["Smoker"],
        "current_smoker": values["Smoker"],
        "active_smoker": values["Smoker"],
        "acei_arb_usage": values["ACEi_ARB_Usage"],
        "acei_arb": values["ACEi_ARB_Usage"],
        "ace_arb": values["ACEi_ARB_Usage"],
    }
    aliases.update({normalize_name(key): value for key, value in values.items()})
    return {normalize_name(key): float(value) for key, value in aliases.items()}


def build_xgb_frame(inputs: StaticInputs, feature_order: list[str]) -> pd.DataFrame:
    aliases = static_value_aliases(inputs)
    row: dict[str, float] = {}
    missing: list[str] = []
    for feature in feature_order:
        normalized = normalize_name(feature)
        if normalized in aliases:
            row[feature] = aliases[normalized]
        else:
            missing.append(feature)
    if missing:
        raise ValueError("The XGBoost model expects unsupported features: " + ", ".join(missing))
    return pd.DataFrame([row], columns=feature_order)


def predict_xgb_probability(model: Any, inputs: StaticInputs, feature_order: list[str]) -> float:
    frame = build_xgb_frame(inputs, feature_order)
    if hasattr(model, "predict_proba"):
        return coerce_probability(model.predict_proba(frame))
    try:
        import xgboost as xgb

        dmatrix = xgb.DMatrix(frame, feature_names=feature_order)
        return coerce_probability(model.predict(dmatrix))
    except Exception:
        return coerce_probability(model.predict(frame))


def validate_inputs(inputs: StaticInputs, uacr_values: list[float]) -> list[str]:
    errors: list[str] = []
    if inputs.dob > date.today():
        errors.append("Date of birth cannot be in the future.")
    if not (AGE_MIN <= inputs.age <= AGE_MAX):
        errors.append(f"Date of birth must correspond to an age between {AGE_MIN:.0f} and {AGE_MAX:.0f} years.")
    if not (WEIGHT_MIN <= inputs.weight_kg <= WEIGHT_MAX):
        errors.append(f"Weight must be between {WEIGHT_MIN:.1f} and {WEIGHT_MAX:.1f} kg.")
    if not (HEIGHT_MIN <= inputs.height_cm <= HEIGHT_MAX):
        errors.append(f"Height must be between {HEIGHT_MIN:.1f} and {HEIGHT_MAX:.1f} cm.")
    if not (BMI_MIN <= inputs.bmi <= BMI_MAX):
        errors.append("The height and weight combination is outside the model's supported range.")
    if not (SBP_MIN <= inputs.baseline_systolic_bp <= SBP_MAX):
        errors.append(f"Baseline systolic BP must be between {SBP_MIN} and {SBP_MAX} mmHg.")
    if len(uacr_values) != WEEKS_IN_SERIES:
        errors.append("Exactly 12 weekly UACR readings are required.")
    for index, value in enumerate(uacr_values, start=1):
        if not np.isfinite(value):
            errors.append(f"Week {index} UACR must be finite.")
        elif not (UACR_MIN <= value <= UACR_MAX):
            errors.append(
                f"Week {index} UACR must be between {UACR_MIN:.1f} and {UACR_MAX:.1f} mg/g."
            )
    return errors


def effective_savgol_window(length: int) -> int:
    window = min(SAVGOL_WINDOW_LENGTH, length if length % 2 == 1 else length - 1)
    minimum = SAVGOL_POLY_ORDER + 2
    if minimum % 2 == 0:
        minimum += 1
    window = max(window, minimum)
    if window > length:
        window = length if length % 2 == 1 else length - 1
    return max(window, 1)


def smooth_uacr_values(values: list[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32)
    if savgol_filter is None:
        return raw.copy()
    window = effective_savgol_window(len(raw))
    if window <= SAVGOL_POLY_ORDER:
        return raw.copy()
    smoothed = savgol_filter(
        raw,
        window_length=window,
        polyorder=SAVGOL_POLY_ORDER,
        mode="nearest",
    )
    return np.asarray(smoothed, dtype=np.float32)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    denominator = float(np.sum(x_centered * x_centered))
    if denominator <= 0:
        return 0.0
    return float(np.sum(x_centered * (y - np.mean(y))) / denominator)


def process_uacr(values: list[float]) -> UACRProcessing:
    raw = np.asarray(values, dtype=np.float32)
    smoothed = smooth_uacr_values(values)
    smoothing_delta = smoothed - raw
    smoothing_delta_max = float(np.max(np.abs(smoothing_delta))) if smoothing_delta.size else 0.0
    mean = float(np.mean(smoothed))
    std = float(np.std(smoothed, ddof=1))
    if not np.isfinite(std) or std < 1e-6:
        std = 1e-6
    z_scores = (smoothed - mean) / std
    weeks = np.arange(1, WEEKS_IN_SERIES + 1, dtype=np.float32)
    beta1 = ols_slope(weeks, z_scores)
    raw_slope = ols_slope(weeks, smoothed)

    centered = weeks - float(np.mean(weeks))
    centered_norm = centered / max(float(np.max(np.abs(centered))), 1.0)
    trend_strength = float(np.clip(abs(beta1), 0.0, 1.25))
    raw_weights = np.exp(trend_strength * centered_norm)
    weights = raw_weights / float(np.sum(raw_weights))
    uacr_opt = float(np.sum(weights * smoothed))

    return UACRProcessing(
        raw=raw,
        smoothed=smoothed,
        smoothing_delta=np.asarray(smoothing_delta, dtype=np.float32),
        smoothing_delta_max=smoothing_delta_max,
        z_scores=np.asarray(z_scores, dtype=np.float32),
        beta1=float(beta1),
        raw_slope=float(raw_slope),
        weights=np.asarray(weights, dtype=np.float32),
        uacr_opt=float(np.clip(uacr_opt, UACR_MIN, UACR_MAX)),
        mean=mean,
        std=std,
    )


def reference_profile(frame: pd.DataFrame, age: float, gender: int) -> ReferenceProfile:
    sex_label = "Male" if gender == 1 else "Female"
    column_name = "Male_Median_P50" if gender == 1 else "Female_Median_P50"
    working = frame[["Age", column_name]].copy()
    working["Age"] = pd.to_numeric(working["Age"], errors="coerce")
    working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    working = working.dropna().sort_values("Age")
    if working.empty:
        raise ValueError("Reference CSV contains no usable age/reference rows.")

    min_age = float(working["Age"].min())
    max_age = float(working["Age"].max())
    age_used = float(np.clip(age, min_age, max_age))
    distance = (working["Age"] - age_used).abs()
    nearest_index = distance.idxmin()
    nearest = working.loc[nearest_index]
    reference_value = float(nearest[column_name])
    row_age = float(nearest["Age"])
    curve = np.full(WEEKS_IN_SERIES, reference_value, dtype=np.float32)
    return ReferenceProfile(
        age_used=age_used,
        age_was_clamped=not np.isclose(age_used, age),
        sex_label=sex_label,
        reference_value=reference_value,
        column_name=column_name,
        row_age=row_age,
        curve=curve,
    )


def expected_lstm_shape(model: Any) -> tuple[int, int | None]:
    shape = model_input_shape(model)
    if shape is None or len(shape) < 3:
        return WEEKS_IN_SERIES, 1
    timesteps = int(shape[1]) if shape[1] is not None else WEEKS_IN_SERIES
    features = int(shape[2]) if shape[2] is not None else None
    return timesteps, features


def resize_series(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values.astype(np.float32)
    source_x = np.linspace(1.0, float(len(values)), num=len(values))
    target_x = np.linspace(1.0, float(len(values)), num=length)
    return np.interp(target_x, source_x, values).astype(np.float32)


def minmax_scale_lstm_uacr(values: np.ndarray) -> np.ndarray:
    resized_min = resize_series(LSTM_UACR_MIN, len(values))
    resized_max = resize_series(LSTM_UACR_MAX, len(values))
    denominator = np.maximum(resized_max - resized_min, 1e-6)
    return ((values.astype(np.float32) - resized_min) / denominator).astype(np.float32)


def build_lstm_tensor(model: Any, uacr_processing: UACRProcessing) -> np.ndarray:
    timesteps, feature_count = expected_lstm_shape(model)
    smoothed = resize_series(uacr_processing.smoothed, timesteps)

    if feature_count is None or feature_count == 1:
        log_uacr = np.log1p(np.clip(smoothed, 0.0, UACR_MAX))
        return log_uacr.reshape(1, timesteps, 1).astype(np.float32)

    scaled_uacr = minmax_scale_lstm_uacr(smoothed)

    if feature_count == WEEKS_IN_SERIES:
        feature_vector = resize_series(scaled_uacr, WEEKS_IN_SERIES)
    elif feature_count == WEEKS_IN_SERIES + 1:
        feature_vector = np.concatenate(([0.5], resize_series(scaled_uacr, WEEKS_IN_SERIES)))
    else:
        base = np.concatenate(([0.5], resize_series(scaled_uacr, min(WEEKS_IN_SERIES, feature_count - 1))))
        if len(base) < feature_count:
            base = np.pad(base, (0, feature_count - len(base)), constant_values=0.0)
        feature_vector = base[:feature_count]

    tensor = np.repeat(feature_vector.reshape(1, 1, feature_count), timesteps, axis=1)
    return tensor.astype(np.float32)


def predict_trend_probability(model: Any, uacr_processing: UACRProcessing) -> tuple[float, float, tuple[int, ...], str]:
    tensor = build_lstm_tensor(model, uacr_processing)
    raw_prediction = model.predict(tensor, verbose=0)
    model_probability = coerce_probability(raw_prediction)
    return (
        model_probability,
        model_probability,
        tuple(int(dim) for dim in tensor.shape),
        "Trained LSTM Ptrend used directly.",
    )


def albuminuria_category(uacr_opt: float) -> tuple[str, str]:
    if uacr_opt >= MACROALBUMINURIA_MIN:
        return (
            "A3",
            "A3 severely increased albuminuria.",
        )
    if uacr_opt >= MICROALBUMINURIA_MIN:
        return (
            "A2",
            "A2 moderately increased albuminuria.",
        )
    return (
        "A1",
        "A1 normal-to-mildly increased albuminuria.",
    )


def smooth_score(value: float, midpoint: float, scale: float) -> float:
    scale = max(float(scale), 1e-6)
    z = float(np.clip((value - midpoint) / scale, -60.0, 60.0))
    return float(1.0 / (1.0 + math.exp(-z)))


def uacr_burden_score(uacr_opt: float) -> float:
    uacr = float(np.clip(uacr_opt, UACR_MIN, UACR_MAX))
    if uacr < 10.0:
        return 0.05 + 0.05 * (uacr / 10.0)
    if uacr < MICROALBUMINURIA_MIN:
        return 0.10 + 0.22 * ((uacr - 10.0) / 20.0)
    if uacr < MACROALBUMINURIA_MIN:
        return 0.45 + 0.25 * ((uacr - MICROALBUMINURIA_MIN) / 270.0)
    return 0.82 + 0.12 * smooth_score(math.log1p(uacr), math.log1p(MACROALBUMINURIA_MIN), 1.25)


def dynamic_late_fusion(
    p_xgb: float,
    p_trend: float,
    uacr_processing: UACRProcessing,
) -> tuple[float, float, float, float, str]:
    """
    Evidence-adaptive calibrated logit stacking.

    The starting weight is anchored to model discrimination above chance
    (AUC-0.5). It is then shifted by serial UACR evidence: KDIGO albuminuria
    burden, relative ACR change, internal z-slope, and fractional weekly slope.
    KDIGO treats a doubling of ACR as exceeding ordinary laboratory variability.
    """
    category, category_note = albuminuria_category(uacr_processing.uacr_opt)
    early_uacr = max(float(np.median(uacr_processing.smoothed[:3])), UACR_MIN)
    recent_uacr = max(float(np.median(uacr_processing.smoothed[-3:])), UACR_MIN)
    relative_change = recent_uacr / early_uacr
    log2_change = float(np.log2(max(relative_change, 1e-6)))

    burden = uacr_burden_score(uacr_processing.uacr_opt)
    increase_score = smooth_score(log2_change, math.log2(1.50), 0.35)
    z_rise_score = smooth_score(uacr_processing.beta1, 0.10, 0.08)
    fractional_weekly_slope = uacr_processing.raw_slope / max(uacr_processing.uacr_opt, 10.0)
    slope_score = smooth_score(fractional_weekly_slope, 0.025, 0.018)
    trajectory_evidence = float(
        np.clip(
            0.45 * burden
            + 0.25 * increase_score
            + 0.20 * z_rise_score
            + 0.10 * slope_score,
            0.0,
            1.0,
        )
    )

    stable_a1 = (
        category == "A1"
        and 0.75 <= relative_change <= 1.25
        and abs(uacr_processing.beta1) < 0.10
        and p_trend < 0.15
    )
    improving = relative_change <= 0.70 and uacr_processing.beta1 < -0.10
    serial_reassurance = 1.0 if stable_a1 else 0.0

    trend_weight = (
        MODEL_TREND_RELIABILITY_WEIGHT
        + 0.55 * (trajectory_evidence - 0.50)
        + 0.12 * serial_reassurance
    )
    if improving:
        trend_weight -= 0.08
    if relative_change >= 2.0:
        trend_weight = max(trend_weight, 0.62)
    if category == "A3":
        trend_weight = max(trend_weight, 0.66)
    if stable_a1 and p_xgb >= 0.50:
        trend_weight = max(trend_weight, 0.58)
    if category == "A1" and relative_change < 1.35 and uacr_processing.beta1 < 0.12:
        trend_weight = min(trend_weight, 0.60)

    trend_weight = float(np.clip(trend_weight, DYNAMIC_TREND_WEIGHT_MIN, DYNAMIC_TREND_WEIGHT_MAX))
    xgb_weight = float(1.0 - trend_weight)
    logit_value = META_INTERCEPT + xgb_weight * safe_logit(p_xgb) + trend_weight * safe_logit(p_trend)
    fused = float(np.clip(sigmoid(logit_value), 0.0, 1.0))
    note = (
        f"Evidence-adaptive fusion: {category} ({category_note}) | "
        f"3-week median UACR change {relative_change:.2f}x | "
        f"trajectory evidence {trajectory_evidence:.2f} "
        f"(burden {burden:.2f}, rise {increase_score:.2f}, z-slope {z_rise_score:.2f}, "
        f"fractional slope {slope_score:.2f}) | "
        f"model-reliability anchor LSTM {MODEL_TREND_RELIABILITY_WEIGHT:.2f} | "
        f"final weights: XGB {xgb_weight:.2f}, LSTM {trend_weight:.2f}."
    )
    return fused, xgb_weight, trend_weight, float(relative_change), note


def apply_severity_floor(probability: float, uacr_opt: float) -> tuple[float, str, bool]:
    if uacr_opt >= MACROALBUMINURIA_MIN:
        floored = max(probability, MACRO_SEVERITY_FLOOR)
        return floored, "A3 macroalbuminuria floor: final risk cannot be below 28.0%.", floored != probability
    if MICROALBUMINURIA_MIN <= uacr_opt < MACROALBUMINURIA_MIN:
        floored = max(probability, MICRO_SEVERITY_FLOOR)
        return floored, "A2 microalbuminuria floor: final risk cannot be below 11.0%.", floored != probability
    return probability, "No albuminuria severity floor activated.", False


def stratify_risk(percent: float) -> str:
    if percent < LOW_RISK_MAX:
        return "Low Risk"
    if percent < MODERATE_RISK_MAX:
        return "Moderate Risk"
    if percent < HIGH_RISK_MAX:
        return "High Risk"
    return "Critical Risk"


def run_prediction(bundle: AssetBundle, inputs: StaticInputs, uacr_values: list[float]) -> PredictionResult:
    if bundle.errors:
        raise RuntimeError("Prediction is unavailable until all required assets load successfully.")
    if bundle.xgb_model is None or bundle.lstm_model is None or bundle.reference_curve is None:
        raise RuntimeError("Prediction is unavailable because an asset is missing.")

    validation_errors = validate_inputs(inputs, uacr_values)
    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    processing = process_uacr(uacr_values)
    reference = reference_profile(bundle.reference_curve, inputs.age, inputs.gender)
    p_xgb = predict_xgb_probability(bundle.xgb_model, inputs, bundle.xgb_feature_names)
    p_trend, p_trend_model_raw, lstm_tensor_shape, trend_probability_note = predict_trend_probability(
        bundle.lstm_model,
        processing,
    )
    fused, xgb_weight, trend_weight, relative_change, fusion_note = dynamic_late_fusion(
        p_xgb,
        p_trend,
        processing,
    )
    final_probability, floor_label, floor_applied = apply_severity_floor(fused, processing.uacr_opt)
    final_percent = final_probability * 100.0
    risk_level = stratify_risk(final_percent)

    return PredictionResult(
        p_xgb=p_xgb,
        p_trend=p_trend,
        p_trend_model_raw=p_trend_model_raw,
        trend_probability_note=trend_probability_note,
        fused_probability=fused,
        xgb_fusion_weight=xgb_weight,
        trend_fusion_weight=trend_weight,
        uacr_relative_change=relative_change,
        fusion_note=fusion_note,
        final_probability=final_probability,
        final_percent=final_percent,
        risk_level=risk_level,
        severity_floor_label=floor_label,
        severity_floor_applied=floor_applied,
        static_feature_order=bundle.xgb_feature_names,
        lstm_tensor_shape=lstm_tensor_shape,
        reference=reference,
        uacr_processing=processing,
    )


def default_dob() -> date:
    today = date.today()
    try:
        return today.replace(year=today.year - 55)
    except ValueError:
        return today.replace(year=today.year - 55, day=28)


def dob_for_age(age_years: float) -> date:
    today = date.today()
    year = today.year - int(age_years)
    try:
        return today.replace(year=year)
    except ValueError:
        return today.replace(year=year, day=28)


WIZARD_STEPS = (("Patient", "Patient details"), ("History", "Clinical history"), ("UACR", "Weekly UACR"), ("Review", "Review assessment"))


WIZARD_DATA_KEYS = (
    "patient_name",
    "patient_dob",
    "patient_sex",
    "patient_weight_kg",
    "patient_height_cm",
    "baseline_sbp",
    "has_diabetes",
    "has_hypertension",
    "is_smoker",
    "uses_acei_arb",
    *(f"uacr_week_{index}" for index in range(1, WEEKS_IN_SERIES + 1)),
)


def initialize_session_state() -> None:
    defaults: dict[str, Any] = {
        "wizard_step": 1,
        "wizard_schema_version": 3,
        "patient_name": "",
        "patient_dob": default_dob(),
        "patient_sex": None,
        "patient_weight_kg": 78.0,
        "patient_height_cm": 170.0,
        "baseline_sbp": 132,
        "has_diabetes": None,
        "has_hypertension": None,
        "is_smoker": None,
        "uses_acei_arb": None,
        "review_confirmed": False,
        "analysis_result": None,
        "submitted_snapshot": None,
        "analysis_error": "",
        "step_errors": [],
    }
    defaults.update({f"uacr_week_{index}": 10.0 for index in range(1, WEEKS_IN_SERIES + 1)})
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Migrate values from the earlier boolean-based wizard without losing data.
    for key in ("has_diabetes", "has_hypertension", "is_smoker", "uses_acei_arb"):
        if isinstance(st.session_state.get(key), bool):
            st.session_state[key] = "Yes" if st.session_state[key] else "No"

    # Old completed results do not include the name needed by the new greeting.
    snapshot = st.session_state.get("submitted_snapshot")
    if isinstance(snapshot, dict) and "patient_name" not in snapshot:
        st.session_state["analysis_result"] = None
        st.session_state["submitted_snapshot"] = None
        st.session_state["wizard_step"] = 1

    # Streamlit removes widget-backed keys when their widget disappears.
    # Reassignment keeps answers durable across the assessment screens.
    for key in WIZARD_DATA_KEYS:
        st.session_state[key] = st.session_state[key]

    try:
        current_step = int(st.session_state["wizard_step"])
    except (TypeError, ValueError):
        current_step = 1
    st.session_state["wizard_step"] = int(np.clip(current_step, 1, len(WIZARD_STEPS)))
    st.session_state["wizard_schema_version"] = 3


def invalidate_analysis() -> None:
    st.session_state["analysis_result"] = None
    st.session_state["submitted_snapshot"] = None
    st.session_state["analysis_error"] = ""
    st.session_state["step_errors"] = []
    st.session_state["review_confirmed"] = False


def prepare_step_change(step: int) -> None:
    """Update wizard state safely, including when called as a widget callback."""
    st.session_state["wizard_step"] = int(np.clip(step, 1, len(WIZARD_STEPS)))
    st.session_state["step_errors"] = []
    st.session_state["analysis_error"] = ""
    st.session_state["review_confirmed"] = False


def move_to_step(step: int) -> None:
    prepare_step_change(step)
    st.rerun()


def binary_choice_value(key: str) -> int:
    return 1 if st.session_state.get(key) == "Yes" else 0


def inputs_from_state() -> StaticInputs:
    dob_value = st.session_state["patient_dob"]
    if isinstance(dob_value, datetime):
        dob_value = dob_value.date()
    return StaticInputs(
        dob=dob_value,
        age=decimal_age_from_dob(dob_value),
        gender=1 if st.session_state.get("patient_sex") == "Male" else 0,
        height_cm=float(st.session_state["patient_height_cm"]),
        weight_kg=float(st.session_state["patient_weight_kg"]),
        baseline_systolic_bp=int(st.session_state["baseline_sbp"]),
        diabetes=binary_choice_value("has_diabetes"),
        hypertension=binary_choice_value("has_hypertension"),
        smoker=binary_choice_value("is_smoker"),
        acei_arb_usage=binary_choice_value("uses_acei_arb"),
    )


def uacr_values_from_state() -> list[float]:
    return [float(st.session_state[f"uacr_week_{index}"]) for index in range(1, WEEKS_IN_SERIES + 1)]


def state_snapshot(inputs: StaticInputs, uacr_values: list[float]) -> dict[str, Any]:
    return {
        "patient_name": str(st.session_state.get("patient_name", "")).strip(),
        "inputs": inputs,
        "uacr_values": list(uacr_values),
        "created_at": datetime.now(),
    }


def render_error_summary(errors: list[str]) -> None:
    if not errors:
        return
    items = "".join(f"<li>{html.escape(str(error))}</li>" for error in errors)
    st.markdown(
        f"""
        <div class="np-error-summary" role="alert" tabindex="-1" aria-labelledby="np-error-title">
            <strong id="np-error-title">Please check the highlighted fields</strong>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def patient_name_errors() -> list[str]:
    value = str(st.session_state.get("patient_name", "")).strip()
    if not value:
        return ["Enter the patient's name to continue."]
    if len(value) < 2:
        return ["Enter at least two characters for the patient's name."]
    return []


def dob_errors() -> list[str]:
    inputs = inputs_from_state()
    if not (AGE_MIN <= inputs.age <= AGE_MAX):
        return [f"Choose a date of birth for a patient between {AGE_MIN:.0f} and {AGE_MAX:.0f} years old."]
    return []


def measurement_errors() -> list[str]:
    inputs = inputs_from_state()
    if not (BMI_MIN <= inputs.bmi <= BMI_MAX):
        return [
            "This height and weight combination is outside the range supported by the clinical model. "
            "Check both measurements and try again."
        ]
    return []


def choice_errors(key: str, label: str, options: tuple[str, ...] = ("Yes", "No")) -> list[str]:
    return [] if st.session_state.get(key) in options else [f"Select {label} to continue."]


def all_wizard_errors() -> list[str]:
    errors = [
        *patient_name_errors(),
        *dob_errors(),
        *measurement_errors(),
        *choice_errors("patient_sex", "a biological sex", ("Male", "Female")),
        *choice_errors("has_diabetes", "Yes or No for diabetes"),
        *choice_errors("has_hypertension", "Yes or No for hypertension"),
        *choice_errors("is_smoker", "Yes or No for current smoking"),
        *choice_errors("uses_acei_arb", "Yes or No for ACE inhibitor or ARB use"),
    ]
    errors.extend(validate_inputs(inputs_from_state(), uacr_values_from_state()))
    return list(dict.fromkeys(errors))


def out_of_distribution_weeks(values: list[float]) -> list[int]:
    upper_bounds = resize_series(LSTM_UACR_MAX, len(values))
    return [index for index, (value, upper) in enumerate(zip(values, upper_bounds), start=1) if value > float(upper)]


def yes_no(value: int | bool | str) -> str:
    if isinstance(value, str) and value in {"Yes", "No"}:
        return value
    return "Yes" if bool(value) else "No"


def review_row(term: str, value: str) -> str:
    return (
        '<div class="np-review-row">'
        f'<span class="np-review-term">{html.escape(term)}</span>'
        f'<span class="np-review-value">{html.escape(value)}</span>'
        "</div>"
    )


def render_review_cards(inputs: StaticInputs, uacr_values: list[float]) -> None:
    patient_name = str(st.session_state.get("patient_name", "")).strip()
    uacr_pills = "".join(
        f'<span class="np-uacr-pill">Week {index}: {value:.1f}</span>'
        for index, value in enumerate(uacr_values, start=1)
    )
    profile_rows = "".join(
        [
            review_row("Patient name", patient_name),
            review_row("Date of birth", inputs.dob.isoformat()),
            review_row("Biological sex", inputs.sex_label),
            review_row("Height", f"{inputs.height_cm:.1f} cm"),
            review_row("Weight", f"{inputs.weight_kg:.1f} kg"),
        ]
    )
    history_rows = "".join(
        [
            review_row("Systolic BP", f"{inputs.baseline_systolic_bp} mmHg"),
            review_row("Diabetes", yes_no(inputs.diabetes)),
            review_row("Hypertension", yes_no(inputs.hypertension)),
            review_row("Current smoker", yes_no(inputs.smoker)),
            review_row("ACE inhibitor / ARB", yes_no(inputs.acei_arb_usage)),
        ]
    )
    st.markdown(
        f"""
        <div class="np-review-grid">
            <section class="np-review-card" aria-label="Patient profile review">
                <h3>Patient profile</h3><div class="np-review-list">{profile_rows}</div>
            </section>
            <section class="np-review-card" aria-label="Medical history review">
                <h3>Clinical history</h3><div class="np-review-list">{history_rows}</div>
            </section>
            <section class="np-review-card" aria-label="UACR series review">
                <h3>12-week UACR series <span class="np-review-term">mg/g</span></h3>
                <div class="np-uacr-pills">{uacr_pills}</div>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pdf_escape(text: str) -> str:
    return (
        str(text)
        .encode("latin-1", errors="replace")
        .decode("latin-1")
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def rgb_from_hex(hex_color: str) -> tuple[float, float, float]:
    cleaned = hex_color.lstrip("#")
    return tuple(int(cleaned[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def build_simple_pdf_report(
    result: PredictionResult,
    inputs: StaticInputs,
    uacr_values: list[float],
) -> bytes:
    theme = RISK_BANDS[result.risk_level]
    r, g, b = rgb_from_hex(theme["color"])
    lines = [
        "NephroPreempt - Predictive Clinical Analytics Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Patient Profile",
        f"Date of birth: {inputs.dob.isoformat()}",
        f"Sex: {inputs.sex_label} (Gender={inputs.gender})",
        f"Height: {inputs.height_cm:.1f} cm",
        f"Weight: {inputs.weight_kg:.1f} kg",
        f"Baseline systolic BP: {inputs.baseline_systolic_bp} mmHg",
        f"Diabetes: {yes_no(inputs.diabetes)}",
        f"Hypertension: {yes_no(inputs.hypertension)}",
        f"Current smoker: {yes_no(inputs.smoker)}",
        f"ACEi/ARB usage: {yes_no(inputs.acei_arb_usage)}",
        "",
        "Kinetic UACR Data",
        "Raw weekly UACR: " + ", ".join(f"{value:.1f}" for value in uacr_values),
        "Smoothed UACR: " + ", ".join(f"{value:.1f}" for value in result.uacr_processing.smoothed),
        f"UACR_opt: {result.uacr_processing.uacr_opt:.2f} mg/g",
        f"Internal Z-slope beta1: {result.uacr_processing.beta1:.4f}",
        f"Raw OLS slope: {result.uacr_processing.raw_slope:.4f} mg/g/week",
        "",
        "Model Outputs",
        f"Static XGBoost risk Pxgb: {result.p_xgb * 100.0:.2f}%",
        f"Trained LSTM Ptrend used in fusion: {result.p_trend * 100.0:.2f}%",
        f"LSTM note: {result.trend_probability_note}",
        f"Dynamic fusion risk before floor: {result.fused_probability * 100.0:.2f}%",
        f"XGBoost fusion weight: {result.xgb_fusion_weight:.2f}",
        f"LSTM fusion weight: {result.trend_fusion_weight:.2f}",
        f"UACR relative change: {result.uacr_relative_change:.2f}x",
        f"Fusion note: {result.fusion_note}",
        result.severity_floor_label,
        "",
        f"FINAL RISK: {result.final_percent:.2f}%",
        f"RISK LEVEL: {result.risk_level}",
        "",
        "Clinical Directive",
        theme["directive"],
        "",
        "Reference Curve Verification",
        f"Reference file: {REFERENCE_CURVE_PATH.name}",
        f"Reference column: {result.reference.column_name}",
        f"Healthy P50 baseline: {result.reference.reference_value:.2f} mg/g",
        "",
        "System Footnote Verification",
        f"This report was evaluated using {XGB_MODEL_PATH.name}, "
        f"{LSTM_MODEL_PATH.name}, and {REFERENCE_CURVE_PATH.name} with "
        "absolute kidney-safety floors for early warning.",
        "Clinical decision-support only. Interpret with the complete patient record and local care protocols; "
        "do not use this report as a diagnosis or for emergency triage.",
    ]

    page_width, page_height = 612, 792
    left_margin = 48
    top_y = 742
    line_height = 15
    pages: list[str] = []
    current: list[str] = []
    y = top_y

    def flush_page() -> None:
        nonlocal current, y
        pages.append("\n".join(current))
        current = []
        y = top_y

    headings = {
        "Patient Profile",
        "Kinetic UACR Data",
        "Model Outputs",
        "Clinical Directive",
        "Reference Curve Verification",
        "System Footnote Verification",
    }
    for line in lines:
        wrapped_lines = (
            [line]
            if not line or line in headings or line.startswith("FINAL RISK:")
            else textwrap.wrap(str(line), width=92, break_long_words=False, break_on_hyphens=False) or [""]
        )
        for wrapped_line in wrapped_lines:
            if y < 60:
                flush_page()
            if wrapped_line.startswith("FINAL RISK:"):
                current.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
                current.append(f"42 {y - 10:.1f} 528 28 re f")
                current.append("1 1 1 rg")
                current.append(f"BT /F2 16 Tf {left_margin} {y:.1f} Td ({pdf_escape(wrapped_line)}) Tj ET")
                current.append("0 0 0 rg")
                y -= 32
                continue
            font = "/F2 13 Tf" if wrapped_line in headings else "/F1 10 Tf"
            current.append(f"BT {font} {left_margin} {y:.1f} Td ({pdf_escape(wrapped_line)}) Tj ET")
            y -= line_height if wrapped_line else 8
    flush_page()

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_object_numbers: list[int] = []
    for page_content in pages:
        page_num = len(objects) + 1
        content_num = page_num + 1
        page_object_numbers.append(page_num)
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            f"/Contents {content_num} 0 R >>"
        )
        stream = page_content.encode("latin-1", errors="replace")
        content_obj = b"<< /Length " + str(len(stream)).encode("latin-1") + b" >>\nstream\n" + stream + b"\nendstream"
        objects.append(page_obj.encode("latin-1"))
        objects.append(content_obj)

    # Rebuild the pages object now that object numbers are known.
    kids = " ".join(f"{num} 0 R" for num in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode("latin-1")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(pdf)


# Presentation layer. Clinical/model/PDF functions above are preserved verbatim.

def configure_page() -> None:
    st.set_page_config(page_title="NephroPreempt | Kidney risk assessment", page_icon=PROJECT_LOGO_DATA_URI,
                       layout="wide", initial_sidebar_state="collapsed")
    # One design layer, scoped to this app's native components. A single light
    # clinical theme replaces the former conflicting light/dark overrides.
    st.html('''<style>
    .stApp {
        --np-bg:#f5f7f7; --np-surface:#ffffff; --np-ink:#172e32;
        --np-muted:#52666a; --np-accent:#08685e; --np-accent-hover:#06574e;
        --np-tint:#eaf4f1; --np-line:#d9e3e2; --np-input-line:#849995;
        --np-error:#a12828; --np-error-bg:#fff1ef;
        --np-radius:12px; --np-radius-sm:8px; --np-shadow:0 2px 8px #172e3205;
        --np-s1:4px; --np-s2:8px; --np-s3:12px; --np-s4:16px; --np-s6:24px; --np-s8:32px;
        --np-fast:160ms; --np-enter:240ms; --np-ease:cubic-bezier(.2,.7,.3,1);
        color:var(--np-ink); background:var(--np-bg); color-scheme:light;
        --np-font:"Source Sans",sans-serif;
        font-family:var(--np-font);
    }
    [data-testid="stHeader"] {display:none;}
    [data-testid="stMainBlockContainer"] {max-width:960px; padding:32px 24px 40px;}
    .st-key-np_app {min-width:0;}
    .st-key-np_app [data-testid="stVerticalBlock"] {gap:16px;}
    .st-key-np_app :is(h1,h2,h3,p,label,li) {color:var(--np-ink); overflow-wrap:anywhere;}
    .st-key-np_app :is(h1,h2,h3,p,label,input,button,table) {font-family:var(--np-font);}
    .st-key-np_app :is(h1,h2,h3) {font-family:inherit; text-wrap:balance; letter-spacing:-.025em;}
    .st-key-np_app :is(p,label) {font-size:16px; line-height:1.5;}
    .st-key-np_app h2 {font-size:28px; font-weight:650; padding:0; margin:0; line-height:1.2;}
    .st-key-np_app h3 {font-size:19px; font-weight:650; padding:0; margin:0 0 4px;}
    .st-key-np_app a {color:var(--np-accent);}
    .np-header {display:flex; align-items:center; gap:12px; padding-bottom:20px; border-bottom:1px solid var(--np-line);}
    .np-header img {width:64px; height:44px; object-fit:contain; flex-shrink:0;}
    .np-header h1 {font-size:24px; font-weight:700; line-height:1.2; margin:0; padding:0;}
    .np-header p {font-size:14px; color:var(--np-muted); margin:4px 0 0;}
    .st-key-np_app .np-progress {display:flex; gap:6px; margin:6px 0 4px; padding:0; list-style:none;}
    .np-progress li {min-width:0; flex:1; border-top:3px solid var(--np-line); padding:10px 0 0;
        font-size:13px; color:var(--np-muted); line-height:1.4;}
    .np-progress li.is-current {border-color:var(--np-accent); color:var(--np-accent); font-weight:700;}
    .np-progress li.is-done {border-color:var(--np-accent);}
    .np-heading {margin:4px 0 0;}
    .np-heading p {margin:8px 0 0; color:var(--np-muted);}
    .st-key-np_app :is(.st-key-np_profile,.st-key-np_history,.st-key-np_weeks) {
        background:var(--np-surface); border:1px solid var(--np-line); border-radius:var(--np-radius);
        padding:24px; box-shadow:var(--np-shadow);
    }
    .st-key-np_app [data-testid="stWidgetLabel"] {min-height:24px; margin-bottom:4px;}
    .st-key-np_app [data-testid="stWidgetLabel"] p {font-size:15px; font-weight:600;}
    .st-key-np_app :is([data-baseweb="input"],[data-baseweb="base-input"]) {
        background:var(--np-surface); color:var(--np-ink); border-radius:var(--np-radius-sm); min-height:48px;
    }
    .st-key-np_app [data-baseweb="input"] {border:1px solid var(--np-input-line);}
    .st-key-np_app input:not([type="radio"]):not([type="checkbox"]) {min-height:48px; font-size:16px; color:var(--np-ink); caret-color:var(--np-accent);}
    .st-key-np_app [data-testid="stNumberInputContainer"] {height:auto; min-height:50px; border:1px solid var(--np-input-line); border-radius:var(--np-radius-sm);}
    .st-key-np_app [data-testid="stNumberInputContainer"] [data-baseweb="input"] {border:0;}
    .st-key-np_app [data-testid="stNumberInput"] button {min-width:44px; min-height:48px;
        color:var(--np-accent); background:var(--np-surface);}
    .st-key-np_app [data-testid="stNumberInput"] button:hover {background:var(--np-tint);}
    .st-key-np_weeks [data-testid="stNumberInput"] button {display:none;}
    .st-key-np_app [data-baseweb="radio"] {margin:0; min-height:48px; padding:8px 16px 8px 10px;
        border:1px solid var(--np-input-line); border-radius:var(--np-radius-sm); background:var(--np-surface);}
    .st-key-np_app [data-baseweb="radio"]:has(input:checked) {border-color:var(--np-accent); background:var(--np-tint);}
    .st-key-np_app [data-baseweb="radio"] > div:first-child {background:var(--np-accent);}
    .st-key-np_app [role="radiogroup"] {gap:12px; flex-wrap:wrap;}
    .st-key-np_app [data-baseweb="checkbox"] {min-height:48px; align-items:center;}
    .st-key-np_app [data-baseweb="checkbox"] span {color:var(--np-ink);}
    .st-key-np_app :is(.stButton,.stDownloadButton) button {
        min-height:48px; padding:10px 18px; border:1px solid var(--np-input-line);
        border-radius:var(--np-radius-sm); background:var(--np-surface); color:var(--np-ink);
        font-weight:600; box-shadow:none; touch-action:manipulation;
        transition:background-color var(--np-fast) var(--np-ease),border-color var(--np-fast) var(--np-ease),transform 120ms var(--np-ease);
    }
    .st-key-np_app :is(.stButton,.stDownloadButton) button p {color:inherit; font-weight:600;}
    .st-key-np_app :is(.stButton,.stDownloadButton) button:hover {border-color:var(--np-accent); background:var(--np-tint);}
    .st-key-np_app :is(.stButton,.stDownloadButton) button:active {transform:translateY(1px);}
    .st-key-np_app button[kind="primary"] {background:var(--np-accent); border-color:var(--np-accent); color:white;}
    .st-key-np_app button[kind="primary"]:hover {background:var(--np-accent-hover); color:white;}
    .st-key-np_app button:disabled {opacity:.5; cursor:not-allowed; transform:none;}
    .st-key-np_app :is(button,input,a,summary):focus-visible {outline:3px solid var(--np-accent); outline-offset:3px;}
    .st-key-np_app :is([data-baseweb="input"],[data-baseweb="radio"],[data-baseweb="checkbox"]):focus-within {
        outline:3px solid var(--np-accent); outline-offset:3px;
    }
    .st-key-np_app [data-testid="stCaptionContainer"] {opacity:1;}
    .st-key-np_app [data-testid="stCaptionContainer"] p {font-size:14px; color:var(--np-muted); margin:0;}
    .st-key-np_app [data-testid="stExpander"] details {background:var(--np-surface); border:1px solid var(--np-line); border-radius:var(--np-radius-sm);}
    .st-key-np_app [data-testid="stExpander"] summary {min-height:48px; padding:12px 16px; color:var(--np-muted);}
    .st-key-np_app [data-testid="stExpander"] summary p {font-size:14px;}
    .st-key-np_app [data-testid="stAlert"] {border-radius:var(--np-radius-sm);}
    .np-note {font-size:14px; color:var(--np-muted); margin:0; line-height:1.5;}
    .np-inline-error {font-size:14px; color:var(--np-error)!important; margin:0;}
    .np-error-summary {padding:16px; border-left:3px solid var(--np-error); border-radius:var(--np-radius-sm);
        background:var(--np-error-bg); color:var(--np-error);}
    .np-error-summary ul {margin:8px 0 0; padding-left:20px;}
    .np-error-summary li {color:var(--np-error); font-size:14px;}
    .np-review-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px;}
    .np-review-card {min-width:0; background:var(--np-surface); border:1px solid var(--np-line); padding:20px; border-radius:var(--np-radius);}
    .np-review-card:last-child {grid-column:1/-1;}
    .np-review-card h3 {margin-bottom:16px;}
    .np-review-row {display:flex; justify-content:space-between; gap:16px; padding:8px 0; border-bottom:1px solid var(--np-line);}
    .np-review-row:last-child {border:0;}
    .np-review-term {font-size:14px; color:var(--np-muted);}
    .np-review-value {font-size:14px; font-weight:600; text-align:right; overflow-wrap:anywhere; min-width:0;}
    .np-uacr-pills {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px;}
    .np-uacr-pill {font-size:14px; padding:8px; background:var(--np-bg); border-radius:6px; font-variant-numeric:tabular-nums;}
    .np-result {background:var(--np-surface); border:1px solid var(--np-line); border-left:4px solid var(--risk-color);
        border-radius:var(--np-radius); padding:24px; display:grid; grid-template-columns:minmax(180px,.7fr) minmax(0,1.3fr); gap:24px; align-items:center;}
    .np-result.np-new-result {animation:np-result-enter var(--np-enter) var(--np-ease);}
    .np-result-label {margin:0; font-size:14px; color:var(--np-muted);}
    .np-risk-score {font-size:60px; font-weight:650; letter-spacing:-.05em; line-height:1.1; font-variant-numeric:tabular-nums; margin:8px 0;}
    .np-risk-level {font-size:18px; font-weight:700; color:var(--risk-color); margin:0;}
    .np-directive {font-size:16px; margin:0; line-height:1.6;}
    .st-key-np_app .np-legend {display:flex; flex-wrap:wrap; gap:8px 20px; margin:0; padding:0; list-style:none;}
    .np-legend li {font-size:14px; display:flex; align-items:center; gap:8px;}
    .np-legend span {width:20px; border-top:3px solid var(--np-accent);}
    .np-legend .np-raw {border-color:#596d80; border-top-style:dotted;}
    .np-legend .np-reference {border-color:#81672b; border-top-style:dashed;}
    .np-data {width:100%; border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums;}
    .np-data :is(th,td) {padding:8px 4px; text-align:right; border-bottom:1px solid var(--np-line); overflow-wrap:anywhere;}
    .np-data :is(th,td):first-child {text-align:left;}
    .np-data caption {text-align:left; color:var(--np-muted); margin-bottom:8px;}
    .np-footer {padding-top:12px; border-top:1px solid var(--np-line); color:var(--np-muted)!important; font-size:13px!important;}
    .np-clinical-note {padding:16px; border:1px solid var(--np-line); border-radius:var(--np-radius-sm); background:var(--np-tint); margin:0;}
    .st-key-np_focus_helper {display:none;}
    @keyframes np-result-enter {from {opacity:0;transform:translateY(6px)} to {opacity:1;transform:translateY(0)}}
    @media (max-width:640px) {
        [data-testid="stMainBlockContainer"] {padding:20px 16px 32px;}
        .np-header {padding-bottom:16px; gap:8px;}
        .np-header h1 {font-size:22px;}
        .np-header img {width:48px; height:36px;}
        .np-header p {font-size:13px;}
        .st-key-np_app h2 {font-size:26px;}
        .st-key-np_app :is(.st-key-np_profile,.st-key-np_history,.st-key-np_weeks) {padding:16px;}
        .st-key-np_app :is(.st-key-np_profile,.st-key-np_history) [data-testid="stColumn"] {width:100%; flex:1 1 100%; min-width:0;}
        .np-review-grid {grid-template-columns:minmax(0,1fr);}
        .np-uacr-pills {grid-template-columns:repeat(2,minmax(0,1fr));}
        .np-result {grid-template-columns:minmax(0,1fr); gap:16px; padding:20px;}
        .np-risk-score {font-size:52px;}
        .np-progress li {font-size:12px;}
    }
    .st-key-np_weeks [data-testid="stHorizontalBlock"] {gap:16px;}
    .st-key-np_weeks [data-testid="stColumn"] {min-width:0; width:calc(50% - 8px); flex:1 1 calc(50% - 8px);}
    .st-key-np_actions [data-testid="stHorizontalBlock"] {gap:12px;}
    .st-key-np_actions [data-testid="stColumn"] {min-width:0; width:calc(50% - 6px); flex:1 1 calc(50% - 6px);}
    @media (max-width:379px) {
        .st-key-np_weeks [data-testid="stColumn"] {width:100%; flex-basis:100%;}
        .np-review-row {gap:12px;}
    }
    @media (prefers-reduced-motion:reduce) {
        .st-key-np_app *, .st-key-np_app *::before, .st-key-np_app *::after {animation:none!important; transition:none!important; scroll-behavior:auto!important;}
    }
    </style>''')


def render_header(expanded: bool = False) -> None:
    st.html(f'<header class="np-header"><img src="{PROJECT_LOGO_DATA_URI}" alt="" aria-hidden="true">'
            f'<div><h1>{APP_TITLE}</h1><p>Kidney risk assessment</p></div></header>')


def render_wizard_progress(step: int, complete: bool = False) -> None:
    labels = ("Patient", "History", "UACR", "Review")
    items = []
    for i, label in enumerate(labels, 1):
        cls = "is-current" if i == step else "is-done" if i < step else ""
        current = ' aria-current="step"' if i == step else ''
        items.append(f'<li class="{cls}"{current}>{i}. {label}</li>')
    st.html('<nav aria-label="Assessment steps"><ol class="np-progress">' + ''.join(items) + '</ol></nav>')


def section_heading(title: str, copy: str = "") -> None:
    st.html(f'<div class="np-heading"><h2>{html.escape(title)}</h2>'
            + (f'<p>{html.escape(copy)}</p>' if copy else '') + '</div>')


def inline_errors(errors: list[str]) -> None:
    if st.session_state.get("step_errors"):
        for error in errors:
            st.html(f'<p class="np-inline-error" role="alert">{html.escape(error)}</p>')


def profile_errors() -> list[str]:
    return [*patient_name_errors(), *dob_errors(), *measurement_errors(),
            *choice_errors("patient_sex", "a biological sex", ("Male", "Female"))]


def history_errors() -> list[str]:
    return [error for key, label in (("has_diabetes", "diabetes"), ("has_hypertension", "hypertension"),
            ("is_smoker", "current smoking"), ("uses_acei_arb", "ACE inhibitor or ARB use"))
            for error in choice_errors(key, "Yes or No for " + label)]


def render_navigation(*, step: int, next_label: str, validator: Any | None = None, back_label: str = "Back") -> None:
    with st.container(key="np_actions"):
        if step == 1:
            st.button(next_label, type="primary", width="stretch", key="next_1", on_click=advance_step, args=(2, validator))
        else:
            back, forward = st.columns(2)
            back.button(back_label, width="stretch", key=f"back_{step}", on_click=prepare_step_change, args=(step - 1,))
            forward.button(next_label, type="primary", width="stretch", key=f"next_{step}", on_click=advance_step, args=(step + 1, validator))


def advance_step(step: int, validator: Any | None = None) -> None:
    errors = list(validator() if validator else [])
    if errors:
        st.session_state["step_errors"] = errors
        return
    prepare_step_change(step)


def render_profile_step() -> None:
    section_heading("Patient details", "Enter the patient’s details and current measurements.")
    render_error_summary(st.session_state.get("step_errors", []))
    with st.container(key="np_profile"):
        st.text_input("Patient name", key="patient_name", max_chars=80, placeholder="Patient name", on_change=invalidate_analysis)
        inline_errors(patient_name_errors())
        st.caption("Used in the greeting only. Follow your organization’s privacy policy for identifiers.")
        dob_col, sex_col = st.columns(2, gap="medium")
        with dob_col:
            st.date_input("Date of birth", min_value=dob_for_age(AGE_MAX), max_value=dob_for_age(AGE_MIN),
                          format="YYYY-MM-DD", key="patient_dob", on_change=invalidate_analysis)
            st.caption("Supported ages: 18–90 years.")
            inline_errors(dob_errors())
        with sex_col:
            st.radio("Biological sex at birth", ["Male", "Female"], index=None, horizontal=True,
                     key="patient_sex", on_change=invalidate_analysis)
            inline_errors(choice_errors("patient_sex", "a biological sex", ("Male", "Female")))
        height_col, weight_col = st.columns(2, gap="medium")
        with height_col:
            st.number_input("Height (cm)", min_value=HEIGHT_MIN, max_value=HEIGHT_MAX, step=.1, format="%.1f",
                            key="patient_height_cm", on_change=invalidate_analysis)
        with weight_col:
            st.number_input("Weight (kg)", min_value=WEIGHT_MIN, max_value=WEIGHT_MAX, step=.1, format="%.1f",
                            key="patient_weight_kg", on_change=invalidate_analysis)
        inline_errors(measurement_errors())
    render_navigation(step=1, next_label="Continue to history", validator=profile_errors)


def render_history_step() -> None:
    section_heading("Clinical history", "Use a representative baseline reading and the current medical record.")
    render_error_summary(st.session_state.get("step_errors", []))
    with st.container(key="np_history"):
        st.number_input("Systolic blood pressure (mmHg)", min_value=SBP_MIN, max_value=SBP_MAX, step=1,
                        key="baseline_sbp", on_change=invalidate_analysis)
        st.caption("The upper blood-pressure number; supported range 85–198 mmHg.")
        questions = (("has_diabetes", "Diagnosed diabetes"), ("has_hypertension", "Diagnosed hypertension"),
                     ("is_smoker", "Current active smoker"), ("uses_acei_arb", "Current ACE inhibitor / ARB use"))
        for offset in (0, 2):
            cols = st.columns(2, gap="medium")
            for col, (key, label) in zip(cols, questions[offset:offset+2]):
                with col:
                    st.radio(label, ["Yes", "No"], index=None, horizontal=True, key=key, on_change=invalidate_analysis)
                    inline_errors(choice_errors(key, "Yes or No for " + label.lower()))
        st.caption("Smoking refers to current use. Check ACE inhibitor or angiotensin II receptor blocker use against the medication list.")
    render_navigation(step=2, next_label="Continue to UACR", validator=history_errors)


def render_training_caution(values: list[float]) -> None:
    weeks = out_of_distribution_weeks(values)
    if weeks:
        st.warning(f"Week(s) {', '.join(map(str, weeks))} exceed the corresponding weekly maximum in the model’s training data. "
                   "You can continue, but interpret the result with additional clinical caution.")


def render_uacr_step() -> None:
    section_heading("Weekly UACR", "Enter 12 consecutive weekly readings in mg/g.")
    render_error_summary(st.session_state.get("step_errors", []))
    with st.container(key="np_weeks"):
        st.caption("Week 1 is oldest · Week 12 is newest")
        for first in range(1, 13, 2):
            cols = st.columns(2, gap="medium")
            for col, week in zip(cols, range(first, first+2)):
                with col:
                    st.number_input(f"Week {week} (mg/g)", min_value=UACR_MIN, max_value=UACR_MAX,
                                    step=.1, format="%.1f", key=f"uacr_week_{week}", on_change=invalidate_analysis)
    render_training_caution(uacr_values_from_state())
    render_navigation(step=3, next_label="Review entries", validator=all_wizard_errors)


def edit_assessment(step: int = 1) -> None:
    invalidate_analysis()
    prepare_step_change(step)


def render_review_step(bundle: AssetBundle) -> None:
    section_heading("Review assessment", "Check the entries before running the analysis.")
    errors = all_wizard_errors()
    render_error_summary(errors)
    inputs, values = inputs_from_state(), uacr_values_from_state()
    render_review_cards(inputs, values)
    st.button("Edit patient & history", on_click=prepare_step_change, args=(1,), key="edit_review")
    render_training_caution(values)
    st.checkbox("I have reviewed the entries. The result supports, but does not replace, clinical judgment.", key="review_confirmed")
    st.caption("Do not use this tool for emergency triage.")
    if st.session_state.get("analysis_error"):
        st.error("Analysis could not finish. Try again. If the problem continues, contact the app administrator.")
    with st.container(key="np_actions"):
        back, submit = st.columns(2)
        back.button("Back to UACR", width="stretch", on_click=prepare_step_change, args=(3,), key="back_review")
        clicked = submit.button("Run analysis", type="primary", width="stretch", key="run_analysis",
                                disabled=bool(bundle.errors or errors or not st.session_state["review_confirmed"]))
    if clicked:
        try:
            with st.spinner("Analyzing assessment…"):
                result = run_prediction(bundle, inputs, values)
        except Exception as exc:
            logging.getLogger(__name__).exception("Assessment analysis failed")
            st.session_state["analysis_error"] = str(exc)
            st.rerun()
        st.session_state["analysis_result"] = result
        st.session_state["submitted_snapshot"] = state_snapshot(inputs, values)
        st.session_state["result_just_created"] = True
        st.session_state["analysis_error"] = ""
        st.rerun()


def render_chart(result: PredictionResult) -> None:
    if go is None:
        st.info("Chart unavailable. All readings are in the data table below.")
        render_signal_table(result)
        return
    st.html('<ul class="np-legend" aria-label="Chart legend"><li><span></span>Smoothed</li>'
            '<li><span class="np-raw"></span>Measured</li><li><span class="np-reference"></span>Healthy P50</li></ul>')
    fig = go.Figure()
    weeks = list(range(1, 13))
    for values, name, color, dash, mode in (
        (result.uacr_processing.smoothed, "Smoothed", "#08685e", "solid", "lines+markers"),
        (result.uacr_processing.raw, "Measured", "#596d80", "dot", "lines+markers"),
        (result.reference.curve, "Healthy P50", "#81672b", "dash", "lines"),
    ):
        fig.add_trace(go.Scatter(x=weeks, y=values, name=name, mode=mode,
            line=dict(color=color, width=2.5, dash=dash), marker=dict(size=6),
            hovertemplate="Week %{x}<br>%{y:.1f} mg/g<extra>" + name + "</extra>"))
    fig.update_layout(height=310, template="plotly_white", showlegend=False,
        margin=dict(l=0, r=8, t=8, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans, sans-serif", size=13, color="#52666a"),
        hovermode="closest", dragmode=False, uirevision="nephropreempt-uacr")
    fig.update_xaxes(title="Week", tickmode="array", tickvals=[1, 3, 6, 9, 12], fixedrange=True,
                     gridcolor="#e3eae9", automargin=True)
    fig.update_yaxes(title="UACR (mg/g)", fixedrange=True, rangemode="tozero", gridcolor="#d9e3e2", automargin=True)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar":False, "responsive":True, "scrollZoom":False})


def render_signal_table(result: PredictionResult) -> None:
    rows = ''.join(f'<tr><th scope="row">{i}</th><td>{raw:.1f}</td><td>{smooth:.1f}</td><td>{ref:.1f}</td></tr>'
                   for i, (raw, smooth, ref) in enumerate(zip(result.uacr_processing.raw,
                       result.uacr_processing.smoothed, result.reference.curve), 1))
    st.html('<table class="np-data"><caption>UACR readings · mg/g</caption><thead><tr>'
            '<th scope="col">Week</th><th scope="col">Measured</th><th scope="col">Smoothed</th>'
            '<th scope="col">Healthy P50</th></tr></thead><tbody>' + rows + '</tbody></table>')


def render_result(result: PredictionResult, inputs: StaticInputs, uacr_values: list[float], patient_name: str,
                  generated_at: datetime | None = None) -> None:
    generated_at = generated_at or datetime.now()
    section_heading("Assessment results", f"{patient_name} · {generated_at.strftime('%d %b %Y, %H:%M')}")
    # Contrast-adjusted presentation colors only; RISK_BANDS and PDF remain unchanged.
    colors = {"Low Risk":"#08685e", "Moderate Risk":"#8a5700", "High Risk":"#a63b12", "Critical Risk":"#a12828"}
    animate = " np-new-result" if st.session_state.pop("result_just_created", False) else ""
    st.html(f'<section class="np-result{animate}" style="--risk-color:{colors[result.risk_level]}" aria-label="Risk result">'
            '<div><div class="np-result-label">Estimated kidney risk</div>'
            f'<div class="np-risk-score">{result.final_percent:.1f}%</div>'
            f'<div class="np-risk-level">{html.escape(result.risk_level)}</div></div>'
            f'<p class="np-directive">{html.escape(RISK_BANDS[result.risk_level]["directive"])}</p></section>')
    st.download_button("Download PDF report", data=build_simple_pdf_report(result, inputs, uacr_values),
        file_name=f"NephroPreempt_Report_{generated_at.strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf",
        type="primary", width="stretch", on_click="ignore", key="download_report")
    if result.severity_floor_applied:
        st.warning(result.severity_floor_label)
    render_training_caution(uacr_values)
    st.html('<p class="np-clinical-note" role="note">Clinical decision-support only. Interpret with the complete patient record and local care protocols. '
            'This is not a diagnosis and must not be used for emergency triage.</p>')
    st.markdown("### 12-week UACR trajectory")
    direction = "rising" if result.uacr_processing.raw_slope > .05 else "falling" if result.uacr_processing.raw_slope < -.05 else "stable"
    st.caption(f"{direction.capitalize()} smoothed trend · Optimized UACR {result.uacr_processing.uacr_opt:.1f} mg/g")
    render_chart(result)
    with st.expander("Readings & analysis details"):
        render_signal_table(result)
        st.markdown("### Model outputs")
        metrics = [
            ("Static risk", f"{result.p_xgb*100:.2f}%"), ("Trend risk / raw LSTM output", f"{result.p_trend*100:.2f}% / {result.p_trend_model_raw*100:.2f}%"),
            ("Risk before severity floor", f"{result.fused_probability*100:.2f}%"),
            ("Static / trend weights", f"{result.xgb_fusion_weight:.4f} / {result.trend_fusion_weight:.4f}"),
            ("Recent-to-early UACR change", f"{result.uacr_relative_change:.2f}×"),
            ("Z-score slope", f"{result.uacr_processing.beta1:.4f}"),
            ("OLS slope", f"{result.uacr_processing.raw_slope:.4f} mg/g/week"),
            ("Maximum smoothing delta", f"{result.uacr_processing.smoothing_delta_max:.4f} mg/g"),
            ("LSTM tensor shape", str(result.lstm_tensor_shape)),
            ("Reference match", f"{result.reference.sex_label}, age {result.reference.row_age:g}; {result.reference.reference_value:.2f} mg/g"),
        ]
        st.html(''.join(review_row(label, value) for label, value in metrics))
        st.write(result.severity_floor_label)
        st.caption(result.trend_probability_note)
        st.caption(result.fusion_note)
        if result.uacr_processing.smoothing_delta_max < .01:
            st.caption("Smoothing changed the curve by less than 0.01 mg/g; measured markers and the smoothed line may overlap.")
        with st.expander("Signal trace"):
            st.dataframe(pd.DataFrame({"Week":np.arange(1,13), "Smoothing delta":result.uacr_processing.smoothing_delta,
                "Internal Z":result.uacr_processing.z_scores, "Optimization weight":result.uacr_processing.weights}), hide_index=True, width="stretch")
    st.button("Edit assessment", width="stretch", on_click=edit_assessment, args=(1,), key="edit_result")


def render_system_diagnostics(bundle: AssetBundle) -> None:
    with st.expander("System diagnostics"):
        if bundle.errors:
            st.error("The required models or reference data could not load.")
            for error in bundle.errors:
                st.code(error, wrap_lines=True)
            st.caption("Keep the existing model and CSV files beside app_redesign.py. Check the Python environment has the dependencies named in the error, then reload.")
            if st.button("Reload assets", key="reload_assets"):
                load_assets.clear()
                st.session_state.pop("clinical_asset_bundle", None)
                invalidate_analysis()
                st.rerun()
        else:
            st.caption("Both models and the reference curve loaded. Inference runs in this application.")
        if st.session_state.get("analysis_error"):
            st.code(st.session_state["analysis_error"], wrap_lines=True)
        for warning in bundle.warnings:
            st.warning(warning)
        for label, path in (("XGBoost", XGB_MODEL_PATH), ("LSTM", LSTM_MODEL_PATH), ("Reference", REFERENCE_CURVE_PATH)):
            st.write(f"{label}: {path.name}")
        st.caption("Risk bands: Low <10%; Moderate 10–<25%; High 25–<50%; Critical ≥50%.")
        st.caption("A2: 30 ≤ optimized UACR <300 mg/g, minimum risk 11%. A3: optimized UACR ≥300 mg/g, minimum risk 28%.")
        st.caption("The model supports its trained male/female encoding. Patient names do not enter either model.")
        with st.expander("Model metadata"):
            st.write("XGBoost feature order", bundle.xgb_feature_names)
            st.write("LSTM input shape", bundle.lstm_input_shape)
            st.json(bundle.lstm_config)
            st.caption("Motion: one-time result entrance, 160 ms button feedback; reduced-motion preference respected. No animation components required.")


def snapshot_is_current(snapshot: dict[str, Any]) -> bool:
    return (snapshot.get("patient_name") == str(st.session_state.get("patient_name", "")).strip()
            and vars(snapshot["inputs"]) == vars(inputs_from_state())
            and snapshot.get("uacr_values") == uacr_values_from_state())


def reload_assessment_assets() -> None:
    load_assets.clear()
    st.session_state.pop("clinical_asset_bundle", None)
    invalidate_analysis()


def focus_new_screen(screen: str) -> None:
    """Native st.html supports scripts in Streamlit 1.58; no custom component.

    Focus only on a screen change or a newly submitted error summary, never on
    ordinary field reruns. No user content is interpolated into JavaScript.
    """
    signature = (screen, tuple(st.session_state.get("step_errors", [])))
    previous = st.session_state.get("focus_signature")
    st.session_state["focus_signature"] = signature
    if previous == signature or (previous and previous[0] == screen and not signature[1]):
        return
    sequence = int(st.session_state.get("focus_sequence", 0)) + 1
    st.session_state["focus_sequence"] = sequence
    with st.container(key="np_focus_helper"):
        st.html(f'<script id="np-focus-{sequence}">' + '''
        requestAnimationFrame(() => {
            const root = document.querySelector('.st-key-np_app');
            const heading = root?.querySelector('.np-error-summary') || root?.querySelector('.np-heading h2');
            if (heading) {
                heading.tabIndex = -1;
                heading.focus({preventScroll:true});
                root.closest('[data-testid="stMain"]')?.scrollTo({top:0,behavior:'instant'});
            }
        });
        </script>''', unsafe_allow_javascript=True)


def main() -> None:
    configure_page()
    initialize_session_state()
    snapshot = st.session_state.get("submitted_snapshot")
    if isinstance(snapshot, dict) and not snapshot_is_current(snapshot):
        invalidate_analysis()
    with st.container(key="np_app"):
        render_header()
        if "clinical_asset_bundle" not in st.session_state:
            with st.spinner("Preparing assessment…"):
                st.session_state["clinical_asset_bundle"] = load_assets()
            for error in st.session_state["clinical_asset_bundle"].errors:
                logging.getLogger(__name__).error("Assessment asset error: %s", error)
        bundle = st.session_state["clinical_asset_bundle"]
        if bundle.errors:
            st.error("Analysis is unavailable because required assessment files could not load. Retry loading; if the problem continues, contact the app administrator. You can still enter your assessment.")
            st.button("Retry loading", on_click=reload_assessment_assets, key="retry_loading")
        result, snapshot = st.session_state.get("analysis_result"), st.session_state.get("submitted_snapshot")
        if result is not None and isinstance(snapshot, dict):
            render_result(result, snapshot["inputs"], snapshot["uacr_values"], snapshot["patient_name"], snapshot["created_at"])
        else:
            step = int(st.session_state["wizard_step"])
            render_wizard_progress(step)
            if step == 1:
                render_profile_step()
            elif step == 2:
                render_history_step()
            elif step == 3:
                render_uacr_step()
            else:
                render_review_step(bundle)
        # Diagnostics are deliberately absent from the ordinary patient workflow.
        # Enable only for maintenance with NEPHROPREEMPT_DIAGNOSTICS=1.
        if os.environ.get("NEPHROPREEMPT_DIAGNOSTICS") == "1":
            render_system_diagnostics(bundle)
        focus_new_screen("results" if result is not None else str(st.session_state["wizard_step"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("Assessment interface failed")
        st.error("The assessment could not load. Reload the page; if the problem continues, contact the app administrator.")
