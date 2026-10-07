"""
NephroPreempt Streamlit application.

Single-file clinical decision-support web page that loads the bundled retrained
models and age/sex reference curve from the app folder or Desktop/software.
"""

from __future__ import annotations

import html
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
    from streamlit_lottie import st_lottie
except Exception:  # pragma: no cover - optional visual enhancement
    st_lottie = None

try:
    from streamlit_option_menu import option_menu
except Exception:  # pragma: no cover - optional presentation component
    option_menu = None

try:
    import streamlit_shadcn_ui as shadcn_ui
except Exception:  # pragma: no cover - optional presentation component
    shadcn_ui = None

try:
    from streamlit_extras.metric_cards import style_metric_cards
except Exception:  # pragma: no cover - optional presentation component
    style_metric_cards = None

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


# ---------------------------------------------------------------------------
# Asset contract
# ---------------------------------------------------------------------------

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
    """Return likely Desktop/software locations for local Windows and hosted runtimes."""
    candidates = [
        Path(__file__).resolve().parent,
        Path.home() / "OneDrive" / "Desktop" / "software",
        Path.home() / "Desktop" / "software",
        Path.cwd() / "software",
        Path.cwd().parent / "software",
        Path(__file__).resolve().parent / "software",
        Path("/mnt/c/Users/HP/Desktop/software"),
        Path("/host_mnt/c/Users/HP/Desktop/software"),
    ]

    windows_home = Path("C:/Users/HP/Desktop/software")
    if windows_home not in candidates:
        candidates.append(windows_home)

    return list(dict.fromkeys(candidates))


def directory_has_required_assets(directory: Path) -> bool:
    return all(
        any((directory / filename).exists() for filename in filenames)
        for filenames in ASSET_FILENAME_OPTIONS.values()
    )


def resolve_software_dir() -> Path:
    for directory in candidate_software_dirs():
        if directory_has_required_assets(directory):
            return directory
    return Path.home() / "Desktop" / "software"


def resolve_asset_path(primary_filename: str, preferred_filenames: tuple[str, ...] = ()) -> Path:
    filenames = (*preferred_filenames, primary_filename)
    for filename in filenames:
        for directory in candidate_software_dirs():
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return resolve_software_dir() / primary_filename


def display_path(path: Path) -> str:
    raw = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        return raw
    try:
        windows_path = PureWindowsPath(path)
        return str(windows_path)
    except Exception:
        return raw


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

# The original specification listed window=3/polyorder=2, but that combination
# reproduces each 3-point quadratic almost exactly and therefore looks like no
# smoothing. A 5-point quadratic window is the smallest effective setting for
# the 12-week signal.
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

# MinMaxScaler values inferred from the training data artifact for the bundled
# LSTM. Patient_ID is intentionally neutralized at 0.5 because it is not a
# clinical input in the specification.
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


def configure_page() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} | Kidney Risk Intelligence",
        page_icon="N",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@500;600;700;800&family=Noto+Sans:wght@400;500;600;700&display=swap');

            :root {
                --np-ocean-950: #062d3d;
                --np-ocean-900: #0b3b4e;
                --np-ocean-800: #0f5268;
                --np-ocean-700: #0b7285;
                --np-ocean-600: #0891b2;
                --np-teal-600: #0d9488;
                --np-teal-500: #14b8a6;
                --np-teal-300: #5eead4;
                --np-teal-100: #ccfbf1;
                --np-sky-100: #dff7fb;
                --np-slate-950: #0f2530;
                --np-slate-800: #263f4b;
                --np-slate-600: #526b76;
                --np-slate-500: #667f89;
                --np-line: rgba(64, 137, 155, 0.22);
                --np-line-strong: rgba(8, 145, 178, 0.42);
                --np-card: rgba(255, 255, 255, 0.88);
                --np-card-solid: #ffffff;
                --np-page: #f3fafb;
                --np-danger: #b42318;
                --np-warning: #a15c07;
                --np-success: #087f6a;
                --np-radius-sm: 12px;
                --np-radius-md: 18px;
                --np-radius-lg: 26px;
                --np-shadow-sm: 0 8px 24px rgba(8, 66, 82, 0.07);
                --np-shadow-md: 0 20px 54px rgba(8, 66, 82, 0.12);
                --np-shadow-focus: 0 0 0 4px rgba(8, 145, 178, 0.15), 0 0 30px rgba(20, 184, 166, 0.12);
                --np-ease-out: cubic-bezier(0.16, 1, 0.3, 1);
                --np-ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
                --np-motion-fast: 160ms;
                --np-motion-base: 320ms;
                --np-motion-enter: 440ms;
            }

            /* CSS-only 21st/Magic UI blur-fade translation for Streamlit reruns. */
            @keyframes npStepReveal {
                0% { opacity: 0; transform: translate3d(0, 14px, 0) scale(0.988); filter: blur(5px); }
                65% { opacity: 1; filter: blur(0); }
                100% { opacity: 1; transform: translate3d(0, 0, 0) scale(1); filter: blur(0); }
            }
            @keyframes npCardRise {
                from { opacity: 0; transform: translate3d(0, 10px, 0); }
                to { opacity: 1; transform: translate3d(0, 0, 0); }
            }
            /* Rotating conic beam inspired by animated-gradient border components. */
            @keyframes npBorderBeam {
                to { transform: rotate(1turn); }
            }
            /* Paused sheen cycle keeps the primary action calm rather than continuously flashing. */
            @keyframes npShimmer {
                0%, 68% { transform: translateX(-150%) skewX(-20deg); }
                100% { transform: translateX(260%) skewX(-20deg); }
            }
            @keyframes npAuraFloat {
                0%, 100% { transform: translate3d(0, 0, 0) scale(1); opacity: .48; }
                50% { transform: translate3d(10px, -12px, 0) scale(1.08); opacity: .72; }
            }
            @keyframes npProgressGrow {
                from { transform: scaleX(0); }
                to { transform: scaleX(1); }
            }
            @keyframes npPulseRing {
                0% { transform: scale(.82); opacity: .55; }
                75%, 100% { transform: scale(1.16); opacity: 0; }
            }

            html { scroll-behavior: smooth; }
            body, .stApp, [class*="css"] {
                font-family: "Noto Sans", "Segoe UI", sans-serif;
                color: var(--np-slate-950);
            }
            h1, h2, h3, h4, h5, h6 {
                font-family: "Figtree", "Segoe UI", sans-serif;
                color: var(--np-ocean-950);
                letter-spacing: -0.025em;
                text-wrap: balance;
            }
            p, li { line-height: 1.65; }
            #MainMenu,
            footer,
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"] { display: none !important; }
            header[data-testid="stHeader"] {
                height: .25rem;
                background: transparent;
            }
            [data-testid="stAppViewContainer"] { overflow-x: clip; }
            .stApp {
                color-scheme: light;
                background:
                    radial-gradient(circle at 7% 6%, rgba(34, 211, 238, .14), transparent 28rem),
                    radial-gradient(circle at 94% 14%, rgba(20, 184, 166, .11), transparent 25rem),
                    linear-gradient(180deg, #f8fcfd 0%, var(--np-page) 42%, #f8fbfc 100%);
            }
            .main .block-container {
                width: min(100%, 1180px);
                max-width: 1180px;
                padding: 1.15rem 1.5rem 4rem;
            }
            [data-testid="stSidebar"] { display: none; }

            .np-hero,
            .st-key-hero_shell {
                position: relative;
                isolation: isolate;
                overflow: hidden;
                min-height: 286px;
                margin: .4rem 0 1.35rem;
                padding: clamp(1.5rem, 4vw, 3.35rem);
                border: 1px solid rgba(140, 224, 231, .34);
                border-radius: var(--np-radius-lg);
                background:
                    linear-gradient(132deg, rgba(4, 44, 60, .98), rgba(7, 87, 105, .96) 58%, rgba(13, 148, 136, .90));
                box-shadow: 0 28px 70px rgba(6, 45, 61, .20), inset 0 1px 0 rgba(255,255,255,.18);
            }
            .np-hero::before,
            .st-key-hero_shell::before {
                content: "";
                position: absolute;
                z-index: -2;
                width: 44rem;
                height: 44rem;
                right: -25rem;
                top: -30rem;
                border-radius: 50%;
                background: conic-gradient(from 40deg, transparent 0 42%, rgba(94, 234, 212, .72) 50%, transparent 60% 100%);
                animation: npBorderBeam 12s linear infinite;
            }
            .np-hero::after,
            .st-key-hero_shell::after {
                content: "";
                position: absolute;
                z-index: -1;
                width: 19rem;
                height: 19rem;
                right: -3rem;
                bottom: -9rem;
                border-radius: 50%;
                background: rgba(94, 234, 212, .20);
                filter: blur(42px);
                animation: npAuraFloat 7s var(--np-ease-out) infinite;
            }
            @media (hover: hover) and (pointer: fine) {
                .st-key-hero_shell:hover::before { animation-play-state: paused; }
            }
            .np-brand-row {
                display: flex;
                align-items: center;
                gap: .8rem;
                margin-bottom: 2rem;
            }
            .np-brand-mark {
                display: grid;
                place-items: center;
                width: 42px;
                height: 42px;
                flex: 0 0 42px;
                border: 1px solid rgba(255,255,255,.28);
                border-radius: 13px;
                color: #eafffb;
                background: rgba(255,255,255,.12);
                box-shadow: inset 0 1px 0 rgba(255,255,255,.24), 0 10px 28px rgba(0,0,0,.12);
                backdrop-filter: blur(16px);
            }
            .np-brand-name { color: #f4fffd; font: 700 1.02rem/1.2 "Figtree", sans-serif; }
            .np-brand-meta { color: rgba(230, 252, 250, .72); font-size: .77rem; margin-top: .18rem; }
            .np-eyebrow {
                display: inline-flex;
                align-items: center;
                gap: .5rem;
                margin-bottom: .85rem;
                color: #9ff4e5;
                font: 700 .76rem/1.2 "Figtree", sans-serif;
                letter-spacing: .12em;
                text-transform: uppercase;
            }
            .np-eyebrow-dot { width: 7px; height: 7px; border-radius: 50%; background: #5eead4; box-shadow: 0 0 0 5px rgba(94,234,212,.13); }
            .np-hero h1,
            .st-key-hero_shell h1 {
                max-width: 15ch;
                margin: 0;
                color: #ffffff;
                font-size: clamp(2.35rem, 5vw, 4.5rem);
                line-height: .98;
                letter-spacing: -.052em;
            }
            .np-hero-copy {
                max-width: 61ch;
                margin: 1.05rem 0 0;
                color: rgba(236, 254, 255, .82);
                font-size: clamp(.96rem, 1.5vw, 1.08rem);
            }
            .np-hero-badges { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.35rem; }
            .np-hero-badge {
                display: inline-flex;
                align-items: center;
                min-height: 32px;
                padding: .35rem .68rem;
                border: 1px solid rgba(255,255,255,.16);
                border-radius: 999px;
                color: #eafffb;
                background: rgba(255,255,255,.085);
                font-size: .76rem;
                font-weight: 600;
                backdrop-filter: blur(12px);
            }
            .np-lottie-fallback {
                position: relative;
                display: grid;
                place-items: center;
                width: min(100%, 250px);
                aspect-ratio: 1;
                margin: .5rem auto;
            }
            .np-reduced-only { display: none; }
            .np-lottie-fallback::before,
            .np-lottie-fallback::after {
                content: "";
                position: absolute;
                inset: 16%;
                border: 1px solid rgba(94,234,212,.52);
                border-radius: 50%;
                animation: npPulseRing 2.8s var(--np-ease-out) infinite;
            }
            .np-lottie-fallback::after { animation-delay: 1.35s; }
            .np-vital-card {
                position: relative;
                z-index: 1;
                display: grid;
                place-items: center;
                width: 116px;
                height: 116px;
                border: 1px solid rgba(255,255,255,.24);
                border-radius: 34px;
                color: #ccfbf1;
                background: rgba(255,255,255,.11);
                box-shadow: 0 22px 55px rgba(0,0,0,.16), inset 0 1px 0 rgba(255,255,255,.28);
                backdrop-filter: blur(16px);
            }
            iframe[data-testid="stCustomComponentV1"] {
                display: block !important;
                width: 100% !important;
                border: 0 !important;
                background: transparent !important;
            }

            .np-system-strip {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                flex-wrap: wrap;
                margin: 0 0 1.25rem;
                padding: .82rem 1rem;
                border: 1px solid var(--np-line);
                border-radius: var(--np-radius-sm);
                background: rgba(255,255,255,.68);
                box-shadow: var(--np-shadow-sm);
                backdrop-filter: blur(16px);
            }
            .np-status-copy { color: var(--np-slate-600); font-size: .83rem; }
            .np-status-copy strong { color: var(--np-ocean-950); }
            .np-status-pills { display: flex; flex-wrap: wrap; gap: .45rem; }
            .np-status-pill {
                display: inline-flex;
                align-items: center;
                gap: .42rem;
                min-height: 28px;
                padding: .24rem .58rem;
                border-radius: 999px;
                color: var(--np-ocean-800);
                background: rgba(223,247,251,.82);
                font-size: .73rem;
                font-weight: 700;
            }
            .np-status-pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--np-teal-600); box-shadow: 0 0 0 4px rgba(13,148,136,.10); }
            .np-status-pill.is-error { color: var(--np-danger); background: #fff0ef; }
            .np-status-pill.is-error::before { background: var(--np-danger); }

            .np-progress-card {
                margin-bottom: 1rem;
                padding: 1rem 1.1rem 1.08rem;
                border: 1px solid var(--np-line);
                border-radius: var(--np-radius-md);
                background: var(--np-card);
                box-shadow: var(--np-shadow-sm);
                backdrop-filter: blur(18px);
            }
            .np-progress-top { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
            .np-progress-label { color: var(--np-ocean-950); font-weight: 800; font-size: .9rem; }
            .np-progress-value { color: var(--np-ocean-700); font-size: .8rem; font-variant-numeric: tabular-nums; }
            .np-progress-track { position: relative; overflow: hidden; height: 7px; margin: .75rem 0 .95rem; border-radius: 999px; background: #dcebee; }
            .np-progress-fill {
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, var(--np-ocean-600), var(--np-teal-500));
                box-shadow: 0 0 18px rgba(20,184,166,.34);
                transform-origin: left center;
                animation: npProgressGrow 650ms var(--np-ease-out) both;
            }
            .np-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .5rem; }
            .np-step { display: flex; align-items: center; gap: .5rem; min-width: 0; color: var(--np-slate-500); font-size: .72rem; font-weight: 600; }
            .np-step-index {
                display: grid;
                place-items: center;
                width: 25px;
                height: 25px;
                flex: 0 0 25px;
                border: 1px solid #c8dce1;
                border-radius: 50%;
                background: #f4f9fa;
                color: var(--np-slate-600);
                font: 700 .69rem/1 "Figtree", sans-serif;
            }
            .np-step.is-complete, .np-step.is-current { color: var(--np-ocean-900); }
            .np-step.is-complete .np-step-index { border-color: var(--np-teal-600); color: #fff; background: var(--np-teal-600); }
            .np-step.is-current .np-step-index { border-color: var(--np-ocean-600); color: var(--np-ocean-700); background: #e7f8fb; box-shadow: 0 0 0 4px rgba(8,145,178,.11); }
            .np-step-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

            [class*="st-key-wizard_stage_"] { animation: npStepReveal var(--np-motion-enter) var(--np-ease-out) both; }
            [class*="st-key-wizard_stage_"] [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-result_panel [data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid var(--np-line) !important;
                border-radius: var(--np-radius-md) !important;
                background: var(--np-card) !important;
                box-shadow: var(--np-shadow-md) !important;
                backdrop-filter: blur(18px);
            }
            .np-section-heading { margin-bottom: .3rem; }
            .np-section-kicker { margin-bottom: .42rem; color: var(--np-teal-600); font: 700 .74rem/1.2 "Figtree", sans-serif; letter-spacing: .1em; text-transform: uppercase; }
            .np-section-title { margin: 0; color: var(--np-ocean-950); font: 750 clamp(1.45rem, 3vw, 2rem)/1.18 "Figtree", sans-serif; letter-spacing: -.032em; }
            .np-section-copy { max-width: 66ch; margin: .48rem 0 1.25rem; color: var(--np-slate-600); font-size: .9rem; }
            .np-inline-note {
                margin: .45rem 0 1rem;
                padding: .72rem .82rem;
                border-left: 3px solid var(--np-ocean-600);
                border-radius: 0 10px 10px 0;
                color: var(--np-slate-600);
                background: rgba(223,247,251,.58);
                font-size: .82rem;
                line-height: 1.55;
            }
            .np-nav-hint { margin-top: .8rem; color: var(--np-slate-500); font-size: .76rem; text-align: center; }

            div[data-baseweb="input"],
            div[data-baseweb="base-input"],
            div[data-baseweb="select"] > div,
            [data-testid="stTextArea"] textarea {
                min-height: 48px;
                border: 1px solid #bfd6dc !important;
                border-radius: var(--np-radius-sm) !important;
                background: rgba(255,255,255,.92) !important;
                box-shadow: inset 0 1px 2px rgba(8,66,82,.025);
                transition: border-color var(--np-motion-fast) ease, box-shadow var(--np-motion-fast) ease, background-color var(--np-motion-fast) ease;
            }
            div[data-baseweb="base-input"] { border: 0 !important; border-radius: inherit !important; }
            input,
            textarea,
            [data-testid="stDateInputField"],
            [data-testid="stNumberInputField"] {
                color: var(--np-slate-950) !important;
                -webkit-text-fill-color: var(--np-slate-950) !important;
                caret-color: var(--np-ocean-600) !important;
                background: transparent !important;
                font-family: "Noto Sans", "Segoe UI", sans-serif !important;
            }
            input::placeholder,
            textarea::placeholder { color: #718891 !important; -webkit-text-fill-color: #718891 !important; opacity: 1; }
            div[data-baseweb="select"] div,
            div[data-baseweb="select"] [role="combobox"] {
                color: var(--np-slate-600) !important;
                -webkit-text-fill-color: var(--np-slate-600) !important;
                font-family: "Noto Sans", "Segoe UI", sans-serif !important;
            }
            div[data-baseweb="select"] svg { fill: var(--np-ocean-700) !important; color: var(--np-ocean-700) !important; }
            [role="listbox"] { color: var(--np-slate-950) !important; background: #ffffff !important; }
            [role="option"] { color: var(--np-slate-950) !important; background: #ffffff !important; }
            [role="option"]:hover,
            [role="option"][aria-selected="true"] { background: #e7f8fb !important; }
            [data-testid="stNumberInput"] button {
                border-color: #d2e3e7 !important;
                color: var(--np-ocean-800) !important;
                background: #edf7f8 !important;
            }
            [data-testid="stNumberInput"] button svg,
            [data-testid="stDateInput"] svg { fill: currentColor !important; color: var(--np-ocean-800) !important; }
            [data-testid="stTextArea"] textarea { min-height: 108px; padding: .75rem; }
            div[data-baseweb="input"]:focus-within,
            div[data-baseweb="select"]:focus-within,
            [data-testid="stTextArea"] textarea:focus {
                border-color: var(--np-ocean-600) !important;
                background: #fff !important;
                box-shadow: var(--np-shadow-focus) !important;
            }
            label,
            [data-testid="stWidgetLabel"] p,
            label[data-baseweb="radio"] p,
            [data-testid="stCheckbox"] p,
            [data-testid="stToggle"] p { color: var(--np-slate-800) !important; font-weight: 650 !important; }
            [data-testid="stCaptionContainer"] { color: var(--np-slate-600); }
            [data-baseweb="slider"] [role="slider"] {
                background: var(--np-teal-600) !important;
                box-shadow: 0 0 0 3px rgba(8,145,178,.14);
            }
            [data-baseweb="slider"] [role="slider"] + div,
            [data-testid="stSliderThumbValue"] { color: var(--np-ocean-800) !important; }
            [data-baseweb="slider"] > div > div > div:last-child {
                filter: hue-rotate(174deg) saturate(.72) brightness(.83);
            }
            label[data-baseweb="checkbox"],
            label[data-baseweb="radio"] { min-height: 44px; }
            label[data-baseweb="checkbox"] > div:first-child,
            label[data-baseweb="checkbox"] > span:first-child {
                position: relative;
                border-color: #9fbac2 !important;
                background: #c9dce1 !important;
                box-shadow: inset 0 0 0 1px rgba(6,45,61,.06);
            }
            label[data-baseweb="checkbox"] > div:first-child > div {
                background: #ffffff !important;
                box-shadow: 0 2px 5px rgba(6,45,61,.25);
            }
            label[data-baseweb="checkbox"]:has(input:checked) > div:first-child,
            label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {
                border-color: var(--np-teal-600) !important;
                background: var(--np-teal-600) !important;
            }
            label[data-baseweb="checkbox"]:has(input:checked) > span:first-child::after {
                content: "";
                position: absolute;
                top: 2px;
                left: 5px;
                width: 5px;
                height: 9px;
                border: solid #ffffff;
                border-width: 0 2px 2px 0;
                transform: rotate(45deg);
            }
            label[data-baseweb="checkbox"]:focus-within > div:first-child,
            label[data-baseweb="radio"]:focus-within > div:first-child {
                outline: 3px solid rgba(8,145,178,.36) !important;
                outline-offset: 3px !important;
            }
            label[data-baseweb="radio"] > div:first-child {
                border-color: #8faeb7 !important;
                background: #ffffff !important;
            }
            label[data-baseweb="radio"]:has(input:checked) > div:first-child {
                border-color: var(--np-teal-600) !important;
                background: var(--np-teal-600) !important;
            }

            .stButton > button,
            .stDownloadButton > button,
            [data-testid="stFormSubmitButton"] > button {
                position: relative;
                overflow: hidden;
                width: 100%;
                min-height: 48px;
                border: 1px solid #b6d2d9;
                border-radius: 14px;
                color: var(--np-ocean-900);
                background: rgba(255,255,255,.90);
                box-shadow: 0 7px 18px rgba(8,66,82,.055);
                font-weight: 750;
                touch-action: manipulation;
                cursor: pointer;
                transition: transform var(--np-motion-fast) var(--np-ease-spring), border-color var(--np-motion-fast) ease, box-shadow var(--np-motion-fast) ease, background-color var(--np-motion-fast) ease;
            }
            .stButton > button:hover,
            .stDownloadButton > button:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                transform: translate3d(0, -1px, 0);
                border-color: var(--np-ocean-600);
                color: var(--np-ocean-950);
                box-shadow: 0 12px 24px rgba(8,66,82,.10);
            }
            .stButton > button:active,
            .stDownloadButton > button:active,
            [data-testid="stFormSubmitButton"] > button:active { transform: scale(.985); }
            button[kind="primary"],
            button[kind="primaryFormSubmit"],
            .stDownloadButton > button {
                border-color: transparent !important;
                color: #ffffff !important;
                background: linear-gradient(115deg, var(--np-ocean-700), var(--np-ocean-600) 52%, var(--np-teal-600)) !important;
                box-shadow: 0 12px 28px rgba(8, 145, 178, .22) !important;
            }
            button[kind="primary"]::before,
            button[kind="primaryFormSubmit"]::before,
            .stDownloadButton > button::before {
                content: "";
                position: absolute;
                inset: -30% auto -30% -35%;
                width: 34%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,.34), transparent);
                animation: npShimmer 4.2s ease-in-out infinite;
                pointer-events: none;
            }
            button[kind="primary"] p,
            button[kind="primaryFormSubmit"] p,
            .stDownloadButton > button p { color: #ffffff !important; }
            button:focus-visible,
            input:focus-visible,
            textarea:focus-visible,
            [role="slider"]:focus-visible,
            [role="checkbox"]:focus-visible,
            [role="radio"]:focus-visible {
                outline: 3px solid rgba(8,145,178,.78) !important;
                outline-offset: 3px !important;
            }
            button:disabled { opacity: .48 !important; cursor: not-allowed !important; transform: none !important; box-shadow: none !important; }

            .np-review-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .8rem; margin: .8rem 0 1rem; }
            .np-review-card {
                padding: 1rem;
                border: 1px solid var(--np-line);
                border-radius: var(--np-radius-sm);
                background: linear-gradient(145deg, rgba(255,255,255,.96), rgba(240,249,250,.78));
                box-shadow: var(--np-shadow-sm);
                animation: npCardRise 380ms var(--np-ease-out) both;
            }
            .np-review-card:nth-child(2) { animation-delay: 50ms; }
            .np-review-card:nth-child(3) { animation-delay: 100ms; }
            .np-review-card:nth-child(4) { animation-delay: 150ms; }
            .np-review-card h3 { margin: 0 0 .7rem; font-size: .92rem; }
            .np-review-list { display: grid; gap: .48rem; margin: 0; }
            .np-review-row { display: flex; justify-content: space-between; gap: 1rem; padding-bottom: .42rem; border-bottom: 1px solid rgba(64,137,155,.12); }
            .np-review-row:last-child { padding-bottom: 0; border-bottom: 0; }
            .np-review-term { color: var(--np-slate-600); font-size: .76rem; }
            .np-review-value { color: var(--np-ocean-950); font-size: .79rem; font-weight: 700; text-align: right; overflow-wrap: anywhere; }
            .np-uacr-pills { display: flex; flex-wrap: wrap; gap: .35rem; }
            .np-uacr-pill { padding: .25rem .46rem; border-radius: 8px; color: var(--np-ocean-800); background: #e5f6f8; font-size: .7rem; font-weight: 700; font-variant-numeric: tabular-nums; }
            .np-consent-note { color: var(--np-slate-600); font-size: .8rem; }

            .np-error-summary { margin: 0 0 1rem; padding: .9rem 1rem; border: 1px solid rgba(180,35,24,.24); border-radius: var(--np-radius-sm); color: #7a271a; background: #fff4f2; }
            .np-error-summary strong { display: block; margin-bottom: .35rem; }
            .np-error-summary ul { margin: 0; padding-left: 1.2rem; }
            [data-testid="stAlert"] { border-radius: var(--np-radius-sm); border-width: 1px; }
            [data-testid="stAlert"] p { color: var(--np-ocean-800) !important; line-height: 1.55; }

            .np-result-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin: .75rem 0 1rem; }
            .np-result-head h2 { margin: 0; font-size: clamp(1.7rem, 4vw, 2.6rem); }
            .np-result-head p { margin: .32rem 0 0; color: var(--np-slate-600); }
            .risk-banner {
                position: relative;
                overflow: hidden;
                height: 100%;
                min-height: 280px;
                padding: 1.4rem;
                border: 1px solid rgba(8,66,82,.12);
                border-radius: var(--np-radius-md);
                box-shadow: var(--np-shadow-md);
            }
            .risk-banner::after { content: ""; position: absolute; width: 9rem; height: 9rem; right: -4rem; top: -4rem; border-radius: 50%; background: currentColor; opacity: .06; }
            .risk-kicker { color: var(--np-slate-600); font-size: .73rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
            .risk-score { margin: .65rem 0 .2rem; font: 800 clamp(3.2rem, 8vw, 5.4rem)/.92 "Figtree", sans-serif; letter-spacing: -.06em; font-variant-numeric: tabular-nums; }
            .risk-level { margin-bottom: 1rem; font: 750 1.15rem/1.2 "Figtree", sans-serif; }
            .directive { max-width: 46ch; color: var(--np-slate-800); font-size: .92rem; line-height: 1.62; }
            .metric-panel { height: 100%; padding: 1rem; border: 1px solid var(--np-line); border-radius: var(--np-radius-sm); background: linear-gradient(160deg, #fff, #f2fafb); box-shadow: var(--np-shadow-sm); }
            .small-muted { color: var(--np-slate-600); font-size: .78rem; line-height: 1.55; overflow-wrap: anywhere; }
            div[data-testid="stMetric"] { min-height: 112px; padding: .85rem .9rem; border: 1px solid var(--np-line); border-radius: var(--np-radius-sm); background: linear-gradient(160deg, #fff, #f3fafb); box-shadow: var(--np-shadow-sm); }
            div[data-testid="stMetric"] label { color: var(--np-slate-600) !important; }
            div[data-testid="stMetricValue"] { color: var(--np-ocean-950); font-family: "Figtree", sans-serif; font-weight: 750; font-variant-numeric: tabular-nums; }
            [data-testid="stDataFrame"] { overflow: hidden; border: 1px solid var(--np-line); border-radius: var(--np-radius-sm); }
            [data-testid="stExpander"] { overflow: hidden; border: 1px solid var(--np-line); border-radius: var(--np-radius-sm); background: rgba(255,255,255,.76); box-shadow: var(--np-shadow-sm); }
            .np-footer-note { margin-top: 1.4rem; color: var(--np-slate-500); font-size: .72rem; text-align: center; }

            @media (max-width: 768px) {
                .main .block-container { padding: .7rem .9rem 3rem; }
                .np-hero, .st-key-hero_shell { min-height: auto; border-radius: 22px; }
                .st-key-hero_visual { display: none !important; }
                .np-brand-row { margin-bottom: 1.45rem; }
                .np-hero h1, .st-key-hero_shell h1 { font-size: clamp(2.2rem, 12vw, 3.2rem); }
                .np-progress-card { padding-inline: .8rem; }
                .np-steps { gap: .25rem; }
                .np-step { flex-direction: column; align-items: flex-start; gap: .3rem; }
                .np-step-name { max-width: 100%; font-size: .63rem; white-space: normal; line-height: 1.25; }
                .np-review-grid { grid-template-columns: 1fr; }
                .np-review-card:nth-child(n) { animation-delay: 0ms; }
                .risk-banner { min-height: auto; }
            }
            @media (max-width: 420px) {
                .np-step-name { display: none; }
                .np-progress-top { align-items: flex-start; flex-direction: column; gap: .2rem; }
                .np-review-row { flex-direction: column; gap: .15rem; }
                .np-review-value { text-align: left; }
            }
            @media (prefers-reduced-motion: reduce) {
                html { scroll-behavior: auto; }
                .st-key-motion_visual { display: none !important; }
                .np-reduced-only { display: grid !important; }
                *, *::before, *::after {
                    animation-duration: .01ms !important;
                    animation-iteration-count: 1 !important;
                    scroll-behavior: auto !important;
                    transition-duration: .01ms !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_dark_experience() -> None:
    """Apply the dark NephroPreempt design layer after the base theme."""
    st.markdown(
        """
        <style>
            :root {
                --np-night-1000: #02080b;
                --np-night-950: #030d11;
                --np-night-900: #061419;
                --np-night-850: #081a20;
                --np-night-800: #0b2228;
                --np-night-750: #102c32;
                --np-mint-500: #2dd4bf;
                --np-mint-400: #5eead4;
                --np-mint-300: #8ef4e3;
                --np-cyan-500: #22d3ee;
                --np-ink: #f0fdfa;
                --np-ink-soft: #c9dcdf;
                --np-muted: #91a9af;
                --np-dark-line: rgba(110, 231, 219, .16);
                --np-dark-line-strong: rgba(94, 234, 212, .42);
                --np-dark-card: rgba(8, 26, 32, .86);
                --np-dark-card-solid: #091a20;
                --np-dark-input: #07161b;
                --np-dark-shadow: 0 26px 70px rgba(0, 0, 0, .38);
                --np-dark-glow: 0 0 0 4px rgba(45, 212, 191, .14), 0 0 34px rgba(34, 211, 238, .12);
            }

            html { background: var(--np-night-1000); }
            body,
            .stApp,
            [class*="css"] {
                color: var(--np-ink) !important;
            }
            .stApp {
                color-scheme: dark !important;
                background:
                    radial-gradient(circle at 8% 2%, rgba(8, 145, 178, .18), transparent 30rem),
                    radial-gradient(circle at 93% 12%, rgba(6, 94, 86, .24), transparent 34rem),
                    radial-gradient(circle at 50% 105%, rgba(20, 184, 166, .10), transparent 36rem),
                    linear-gradient(155deg, var(--np-night-1000) 0%, var(--np-night-950) 45%, #041116 100%) !important;
            }
            .main .block-container {
                width: min(100%, 1120px);
                max-width: 1120px;
                padding: 1.1rem 1.5rem 4rem;
            }
            [data-testid="stMainBlockContainer"] {
                width: min(100%, 1120px) !important;
                max-width: 1120px !important;
                margin-inline: auto !important;
                padding: 1.1rem 1.5rem 4rem !important;
            }
            h1, h2, h3, h4, h5, h6,
            [data-testid="stHeadingWithActionElements"] {
                color: var(--np-ink) !important;
            }
            p, li, label, .stCaption, [data-testid="stCaptionContainer"] {
                color: var(--np-ink-soft);
            }
            a { color: var(--np-mint-400); }

            /* Expanded welcome surface: dark, spacious, and logo-led. */
            .st-key-hero_shell {
                min-height: 420px;
                margin: .35rem 0 1.2rem;
                padding: clamp(1.8rem, 4.8vw, 4.2rem);
                border: 1px solid rgba(94, 234, 212, .23);
                border-radius: 32px;
                background:
                    radial-gradient(circle at 78% 35%, rgba(45, 212, 191, .16), transparent 25rem),
                    linear-gradient(135deg, rgba(5, 20, 26, .99), rgba(7, 43, 49, .98) 62%, rgba(6, 94, 86, .82)) !important;
                box-shadow: 0 34px 88px rgba(0, 0, 0, .44), inset 0 1px 0 rgba(255,255,255,.08);
            }
            .st-key-hero_shell::before {
                width: 52rem;
                height: 52rem;
                right: -32rem;
                top: -34rem;
                opacity: .6;
            }
            .np-brand-row { margin-bottom: 2.3rem; }
            .np-brand-mark {
                width: 52px;
                height: 52px;
                flex-basis: 52px;
                border-radius: 17px;
                border-color: rgba(94, 234, 212, .28);
                background: rgba(45, 212, 191, .10);
            }
            .np-brand-name {
                color: var(--np-ink) !important;
                font-size: 1.18rem;
                letter-spacing: -.02em;
            }
            .np-brand-meta { color: rgba(201, 220, 223, .75); font-size: .83rem; }
            .np-hero-project-name {
                max-width: none !important;
                margin: 0 !important;
                color: #f7fffd !important;
                font-size: clamp(3.7rem, 5.4vw, 5.15rem) !important;
                line-height: .9 !important;
                letter-spacing: -.072em !important;
                white-space: nowrap;
                text-shadow: 0 10px 50px rgba(45, 212, 191, .12);
            }
            .np-hero-tagline {
                max-width: 22ch;
                margin: 1.25rem 0 0;
                color: var(--np-mint-300) !important;
                font: 700 clamp(1.25rem, 2.25vw, 1.9rem)/1.15 "Figtree", sans-serif;
                letter-spacing: -.028em;
            }
            .np-hero-copy { color: rgba(218, 241, 240, .78) !important; }
            .np-logo-stage {
                position: relative;
                isolation: isolate;
                display: grid;
                place-items: center;
                width: min(100%, 350px);
                aspect-ratio: 1 / 1;
                min-height: 0;
                margin-inline: auto;
                padding: clamp(1.35rem, 3vw, 2.15rem);
                overflow: visible;
                border: 0;
                border-radius: 50%;
                background: transparent;
                box-shadow: none;
            }
            .np-logo-stage::before,
            .np-logo-stage::after {
                content: "";
                position: absolute;
                border-radius: 50%;
                pointer-events: none;
            }
            .np-logo-stage::before {
                inset: 0;
                z-index: 0;
                background: radial-gradient(
                    circle at 50% 50%,
                    rgba(236,253,250,.98) 0 45%,
                    rgba(153,246,228,.78) 58%,
                    rgba(45,212,191,.34) 73%,
                    rgba(45,212,191,.12) 86%,
                    transparent 100%
                );
                box-shadow: 0 0 44px rgba(45,212,191,.30), 0 0 96px rgba(34,211,238,.15);
            }
            .np-logo-stage::after {
                inset: 1%;
                z-index: 0;
                border: 1px solid rgba(94,234,212,.40);
                animation: npPulseRing 3.4s var(--np-ease-out) infinite;
            }
            .np-project-logo {
                position: relative;
                z-index: 2;
                display: block;
                width: min(82%, 290px);
                height: auto;
                object-fit: contain;
                filter: drop-shadow(0 16px 28px rgba(3, 40, 38, .18));
            }

            /* Compact brand bar used after the welcome screen. */
            .np-compact-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin: .25rem 0 1rem;
                padding: .78rem 1rem;
                border: 1px solid var(--np-dark-line);
                border-radius: 18px;
                background: rgba(7, 22, 27, .78);
                box-shadow: 0 14px 38px rgba(0,0,0,.22);
                backdrop-filter: blur(18px);
            }
            .np-compact-brand { display: flex; align-items: center; gap: .78rem; min-width: 0; }
            .np-compact-logo {
                width: 52px;
                height: 40px;
                object-fit: contain;
                padding: .3rem;
                border-radius: 12px;
                background: rgba(204, 251, 241, .92);
            }
            .np-compact-title { color: var(--np-ink); font: 800 1.08rem/1.1 "Figtree", sans-serif; }
            .np-compact-meta { color: var(--np-muted); font-size: .78rem; margin-top: .18rem; }
            .np-compact-status {
                flex: 0 0 auto;
                padding: .38rem .7rem;
                border: 1px solid rgba(45,212,191,.22);
                border-radius: 999px;
                color: var(--np-mint-300);
                background: rgba(45,212,191,.08);
                font-size: .75rem;
                font-weight: 700;
            }

            /* Progress and phase navigation. */
            .np-progress-card {
                margin: 0 0 1rem;
                padding: 1rem 1.1rem .95rem;
                border: 1px solid var(--np-dark-line);
                border-radius: 20px;
                background: rgba(7, 22, 27, .76);
                box-shadow: 0 18px 48px rgba(0,0,0,.23);
                backdrop-filter: blur(18px);
            }
            .np-progress-label { color: var(--np-ink) !important; }
            .np-progress-value { color: var(--np-muted) !important; }
            .np-progress-track { height: 8px; background: #102b31; }
            .np-progress-fill {
                background: linear-gradient(90deg, #0f766e, var(--np-mint-500), var(--np-cyan-500));
                box-shadow: 0 0 24px rgba(45,212,191,.30);
            }
            .np-progress-milestones {
                display: flex;
                justify-content: space-between;
                gap: .75rem;
                margin-top: .72rem;
                color: var(--np-muted);
                font-size: .72rem;
                font-weight: 600;
            }
            .np-progress-milestones span.is-active { color: var(--np-mint-300); }
            iframe[data-testid="stCustomComponentV1"][title*="streamlit_option_menu"] {
                overflow: hidden !important;
                border-radius: 16px !important;
                background: var(--np-night-900) !important;
            }

            /* Single-question cards. */
            [class*="st-key-wizard_stage_"] {
                animation: npStepReveal var(--np-motion-enter) var(--np-ease-out) both;
            }
            [class*="st-key-question_card_"] [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-review_card_shell [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-result_panel [data-testid="stVerticalBlockBorderWrapper"] {
                position: relative;
                overflow: hidden;
                padding: clamp(1.25rem, 3vw, 2.15rem);
                border: 1px solid var(--np-dark-line-strong) !important;
                border-radius: 26px !important;
                background:
                    linear-gradient(145deg, rgba(12, 36, 42, .90), rgba(5, 18, 23, .94)) !important;
                box-shadow: var(--np-dark-shadow), inset 0 1px 0 rgba(255,255,255,.055);
            }
            [class*="st-key-question_card_"] [data-testid="stVerticalBlockBorderWrapper"]::before,
            .st-key-review_card_shell [data-testid="stVerticalBlockBorderWrapper"]::before {
                content: "";
                position: absolute;
                inset: 0 auto auto 0;
                width: 100%;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(94,234,212,.7), transparent);
            }
            .np-section-heading {
                max-width: 680px;
                margin: .75rem auto 1rem;
                text-align: center;
            }
            .np-section-kicker { color: var(--np-mint-400) !important; letter-spacing: .13em; }
            .np-section-title {
                margin-top: .38rem !important;
                color: var(--np-ink) !important;
                font-size: clamp(2rem, 4.8vw, 3.35rem) !important;
                line-height: 1.02 !important;
            }
            .np-section-copy { max-width: 58ch; margin-inline: auto; color: var(--np-muted) !important; }
            .np-question-number {
                display: inline-grid;
                place-items: center;
                width: 42px;
                height: 42px;
                margin-bottom: .9rem;
                border: 1px solid rgba(94,234,212,.26);
                border-radius: 14px;
                color: var(--np-mint-300);
                background: rgba(45,212,191,.08);
                font: 800 .8rem/1 "Figtree", sans-serif;
            }
            .np-question-hint {
                margin: .9rem 0 .2rem;
                padding: .8rem .9rem;
                border: 1px solid rgba(148,163,184,.14);
                border-radius: 14px;
                color: var(--np-muted) !important;
                background: rgba(2,8,11,.30);
                font-size: .84rem;
            }

            /* Dark native inputs remain the reliable clinical data layer. */
            div[data-baseweb="input"],
            div[data-baseweb="base-input"],
            div[data-baseweb="select"] > div,
            [data-testid="stTextArea"] textarea {
                min-height: 54px !important;
                color: var(--np-ink) !important;
                background: var(--np-dark-input) !important;
                border-color: rgba(148, 210, 211, .22) !important;
                border-radius: 15px !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.025) !important;
            }
            input, textarea,
            div[data-baseweb="input"] input,
            div[data-baseweb="base-input"] input {
                color: var(--np-ink) !important;
                -webkit-text-fill-color: var(--np-ink) !important;
                caret-color: var(--np-mint-400) !important;
            }
            input::placeholder, textarea::placeholder {
                color: #6f8990 !important;
                -webkit-text-fill-color: #6f8990 !important;
                opacity: 1 !important;
            }
            div[data-baseweb="input"]:focus-within,
            div[data-baseweb="base-input"]:focus-within,
            div[data-baseweb="select"] > div:focus-within,
            [data-testid="stTextArea"] textarea:focus {
                border-color: var(--np-mint-400) !important;
                box-shadow: var(--np-dark-glow) !important;
            }
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] label,
            label[data-baseweb="radio"] > div:last-child,
            label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p,
            label[data-baseweb="radio"] p,
            label[data-baseweb="checkbox"] p {
                color: var(--np-ink-soft) !important;
                font-weight: 650;
            }
            [data-testid="stNumberInput"] button {
                min-width: 48px;
                min-height: 48px;
                color: var(--np-mint-300) !important;
                background: #0d282e !important;
                border-color: rgba(94,234,212,.14) !important;
            }
            [role="radiogroup"] {
                gap: .65rem;
            }
            label[data-baseweb="radio"],
            label[data-baseweb="checkbox"] {
                min-height: 48px;
                padding: .62rem .8rem;
                border: 1px solid rgba(148, 210, 211, .14);
                border-radius: 14px;
                background: rgba(3, 13, 17, .38);
                transition: border-color var(--np-motion-fast), background var(--np-motion-fast), transform var(--np-motion-fast);
            }
            label[data-baseweb="radio"]:has(input:checked),
            label[data-baseweb="checkbox"]:has(input:checked) {
                border-color: rgba(94,234,212,.52);
                background: rgba(45,212,191,.10);
            }
            label[data-baseweb="radio"] > div:first-child,
            label[data-baseweb="checkbox"] > span:first-child {
                background: #07161b !important;
                border-color: #527178 !important;
            }
            label[data-baseweb="radio"]:has(input:checked) > div:first-child,
            label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {
                background: var(--np-mint-500) !important;
                border-color: var(--np-mint-400) !important;
            }

            /* Binary questions use the full card width and one centered choice per half. */
            :is(
                .st-key-question_card_has_diabetes,
                .st-key-question_card_has_hypertension,
                .st-key-question_card_is_smoker,
                .st-key-question_card_uses_acei_arb
            ) > [data-testid="stElementContainer"] {
                width: 100% !important;
            }
            :is(
                .st-key-question_card_has_diabetes,
                .st-key-question_card_has_hypertension,
                .st-key-question_card_is_smoker,
                .st-key-question_card_uses_acei_arb
            ) .np-question-number {
                display: grid;
                margin-inline: auto;
            }
            :is(
                .st-key-question_card_has_diabetes,
                .st-key-question_card_has_hypertension,
                .st-key-question_card_is_smoker,
                .st-key-question_card_uses_acei_arb
            ) [data-testid="stWidgetLabel"] {
                justify-content: center !important;
                width: 100%;
                margin: .15rem 0 .9rem;
                text-align: center;
            }
            :is(
                .st-key-question_card_has_diabetes,
                .st-key-question_card_has_hypertension,
                .st-key-question_card_is_smoker,
                .st-key-question_card_uses_acei_arb
            ) [role="radiogroup"] {
                display: grid !important;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                width: 100% !important;
                gap: .9rem !important;
            }
            :is(
                .st-key-question_card_has_diabetes,
                .st-key-question_card_has_hypertension,
                .st-key-question_card_is_smoker,
                .st-key-question_card_uses_acei_arb
            ) [data-testid="stRadio"] {
                width: 100% !important;
            }
            :is(
                .st-key-question_card_has_diabetes,
                .st-key-question_card_has_hypertension,
                .st-key-question_card_is_smoker,
                .st-key-question_card_uses_acei_arb
            ) label[data-baseweb="radio"] {
                justify-content: center;
                width: 100% !important;
                min-height: 76px;
                margin: 0 !important;
                padding: 1rem;
            }
            :is(
                .st-key-question_card_has_diabetes,
                .st-key-question_card_has_hypertension,
                .st-key-question_card_is_smoker,
                .st-key-question_card_uses_acei_arb
            ) label[data-baseweb="radio"] p {
                font-size: 1.04rem;
            }

            /* Weekly UACR labels sit directly beside their inputs without phase boxes. */
            [class*="st-key-uacr_week_"] [data-testid="stNumberInput"] {
                display: grid;
                grid-template-columns: 5.25rem minmax(0, 1fr);
                align-items: center;
                gap: .65rem;
                width: 100%;
            }
            [class*="st-key-uacr_week_"] [data-testid="stWidgetLabel"] {
                align-self: center;
                width: 100%;
                margin: 0 !important;
            }
            [class*="st-key-uacr_week_"] [data-testid="stWidgetLabel"] p {
                white-space: nowrap;
            }
            [class*="st-key-uacr_week_"] div[data-baseweb="input"] {
                min-width: 0;
            }
            [data-testid="stSlider"] [role="slider"] {
                background: var(--np-mint-400) !important;
                box-shadow: 0 0 0 5px rgba(45,212,191,.13);
            }
            [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
                background-color: var(--np-mint-500);
            }

            /* Buttons: high-contrast with stable hover/press feedback. */
            .stButton > button,
            .stDownloadButton > button,
            .stFormSubmitButton > button {
                min-height: 50px;
                border-radius: 15px !important;
                font-weight: 750 !important;
                transition: transform var(--np-motion-fast) var(--np-ease-out), border-color var(--np-motion-fast), box-shadow var(--np-motion-fast), background var(--np-motion-fast);
            }
            .stButton > button[kind="secondary"],
            .stButton > button[data-testid="baseButton-secondary"],
            .stFormSubmitButton > button[kind="secondaryFormSubmit"],
            .stFormSubmitButton > button[data-testid="baseButton-secondaryFormSubmit"],
            [data-testid="stFormSubmitButton"] > button[kind="secondaryFormSubmit"],
            [data-testid="stFormSubmitButton"] > button[data-testid="baseButton-secondaryFormSubmit"],
            .stDownloadButton > button {
                color: var(--np-ink-soft) !important;
                border: 1px solid rgba(148,210,211,.20) !important;
                background: rgba(11,34,40,.76) !important;
            }
            .stFormSubmitButton > button[kind="secondaryFormSubmit"] p,
            .stFormSubmitButton > button[data-testid="baseButton-secondaryFormSubmit"] p,
            [data-testid="stFormSubmitButton"] > button[kind="secondaryFormSubmit"] p,
            [data-testid="stFormSubmitButton"] > button[data-testid="baseButton-secondaryFormSubmit"] p {
                color: var(--np-ink-soft) !important;
            }
            .stButton > button[kind="primary"],
            .stButton > button[data-testid="baseButton-primary"],
            .stFormSubmitButton > button[kind="primaryFormSubmit"],
            .stFormSubmitButton > button[data-testid="baseButton-primaryFormSubmit"] {
                color: #00120f !important;
                border: 1px solid rgba(142,244,227,.64) !important;
                background: linear-gradient(110deg, #5eead4 0%, #2dd4bf 52%, #22d3ee 100%) !important;
                box-shadow: 0 14px 32px rgba(45,212,191,.18);
            }
            .stButton > button:hover:not(:disabled),
            .stDownloadButton > button:hover:not(:disabled),
            .stFormSubmitButton > button:hover:not(:disabled) {
                transform: translateY(-1px);
                border-color: var(--np-mint-300) !important;
                box-shadow: 0 16px 34px rgba(45,212,191,.19);
            }
            .stButton > button:active:not(:disabled),
            .stDownloadButton > button:active:not(:disabled),
            .stFormSubmitButton > button:active:not(:disabled) {
                transform: scale(.985);
            }
            button:focus-visible, input:focus-visible, textarea:focus-visible {
                outline: 3px solid var(--np-mint-300) !important;
                outline-offset: 3px !important;
            }
            button:disabled {
                opacity: .42 !important;
                cursor: not-allowed !important;
            }

            .np-nav-hint,
            .np-inline-note,
            .np-consent-note { color: var(--np-muted) !important; }
            .np-inline-note,
            .np-error-summary,
            [data-testid="stAlert"] {
                border-radius: 16px !important;
                background: rgba(10, 31, 37, .92) !important;
                border-color: var(--np-dark-line) !important;
            }
            [data-testid="stAlert"] p,
            [data-testid="stAlert"] li { color: var(--np-ink-soft) !important; }

            /* Review and result surfaces. */
            .np-result-head p { color: var(--np-ink-soft) !important; }
            .np-review-card {
                border-color: var(--np-dark-line) !important;
                background: rgba(4, 16, 20, .55) !important;
                box-shadow: none !important;
            }
            .np-review-card h3,
            .np-review-value { color: var(--np-ink) !important; }
            .np-review-term { color: var(--np-muted) !important; }
            .np-review-row { border-color: rgba(148,210,211,.10) !important; }
            .np-uacr-pill {
                color: var(--np-mint-300) !important;
                border-color: rgba(94,234,212,.18) !important;
                background: rgba(45,212,191,.08) !important;
            }
            .risk-banner {
                color: var(--np-ink) !important;
                border: 1px solid rgba(94,234,212,.20) !important;
                background: linear-gradient(145deg, rgba(12,39,44,.98), rgba(4,17,21,.98)) !important;
                box-shadow: 0 24px 60px rgba(0,0,0,.28) !important;
            }
            .risk-banner .risk-kicker,
            .risk-banner .directive { color: var(--np-ink-soft) !important; }
            [data-testid="stMetric"] {
                min-height: 118px;
                padding: 1rem !important;
                border: 1px solid var(--np-dark-line) !important;
                border-radius: 18px !important;
                background: rgba(7,24,29,.82) !important;
            }
            [data-testid="stMetricLabel"] p { color: var(--np-muted) !important; }
            [data-testid="stMetricValue"] { color: var(--np-ink) !important; }
            [data-testid="stExpander"] {
                border-color: var(--np-dark-line) !important;
                border-radius: 18px !important;
                background: rgba(6,20,25,.78) !important;
            }
            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] summary p { color: var(--np-ink-soft) !important; }
            [data-testid="stDataFrame"] {
                overflow: hidden;
                border: 1px solid var(--np-dark-line);
                border-radius: 16px;
            }
            .metric-panel,
            .np-system-strip {
                color: var(--np-ink-soft) !important;
                border-color: var(--np-dark-line) !important;
                background: rgba(7,23,28,.76) !important;
            }
            .np-status-copy,
            .np-status-copy strong { color: var(--np-ink-soft) !important; }
            .np-status-pill {
                color: var(--np-mint-300) !important;
                border-color: rgba(94,234,212,.18) !important;
                background: rgba(45,212,191,.07) !important;
            }
            .np-footer-note { color: #6f8990 !important; border-color: rgba(148,210,211,.10) !important; }

            @media (max-width: 768px) {
                .main .block-container { padding: .7rem .85rem 3rem; }
                [data-testid="stMainBlockContainer"] {
                    width: 100% !important;
                    padding: .7rem .85rem 3rem !important;
                }
                .st-key-hero_shell {
                    min-height: auto;
                    padding: 1.25rem;
                    border-radius: 24px;
                }
                .st-key-hero_shell [data-testid="stHorizontalBlock"] { gap: 1rem !important; }
                .np-hero-project-name {
                    font-size: clamp(2.65rem, 11.5vw, 4.15rem) !important;
                    line-height: .9 !important;
                }
                .np-logo-stage {
                    width: min(100%, 280px);
                    min-height: 0;
                    margin-top: .35rem;
                }
                .np-project-logo { width: min(82%, 235px); }
                .np-progress-milestones { font-size: .66rem; }
                .np-compact-status { display: none; }
                [class*="st-key-question_card_"] [data-testid="stVerticalBlockBorderWrapper"],
                .st-key-review_card_shell [data-testid="stVerticalBlockBorderWrapper"] {
                    padding: 1.1rem;
                    border-radius: 20px !important;
                }
                .np-review-grid { grid-template-columns: 1fr !important; }
            }
            @media (max-width: 420px) {
                .st-key-hero_shell { padding: 1rem; }
                .np-brand-row { margin-bottom: 1.15rem; }
                .np-hero-project-name {
                    font-size: clamp(2.35rem, 10.8vw, 3rem) !important;
                    letter-spacing: -.065em !important;
                }
                .np-hero-tagline {
                    margin-top: .8rem;
                    font-size: 1.15rem;
                }
                .np-hero-copy { display: none; }
                .np-hero-badges { display: none; }
                .np-logo-stage {
                    width: min(76vw, 220px);
                    aspect-ratio: 1 / 1;
                    min-height: 0;
                    max-height: none;
                    padding: 1rem;
                    border-radius: 50%;
                }
                .np-project-logo { width: 82%; }
                .np-system-strip,
                .st-key-assessment_phase_indicator { display: none !important; }
                .np-progress-card { padding: .85rem; }
                .np-progress-milestones span:nth-child(2),
                .np-progress-milestones span:nth-child(3) { display: none; }
            }
            @media (prefers-reduced-motion: reduce) {
                .np-logo-stage::before,
                .np-logo-stage::after,
                [class*="st-key-wizard_stage_"] {
                    animation: none !important;
                }
            }

            /* Cloud-safe native widget colors and phone layouts. */
            :root {
                --primary-color: #2dd4bf;
                --background-color: #02080b;
                --secondary-background-color: #081a20;
                --text-color: #f0fdfa;
            }
            html, body, .stApp { max-width: 100%; overflow-x: clip; }
            .np-compact-header { justify-content: flex-start; min-width: 0; }
            .np-compact-logo {
                width: 108px; height: 66px; flex: 0 0 108px;
                padding: .45rem; background: #e8faf6;
                object-fit: contain; border-radius: 12px;
            }
            .np-compact-title { overflow-wrap: anywhere; }
            .np-progress-card { padding-bottom: .55rem; }
            .np-progress-track { margin-bottom: .2rem; }
            .np-section-heading { margin: 1.1rem auto 1rem; }
            .np-section-title { overflow-wrap: anywhere; }
            .np-error-summary {
                padding: .75rem 1rem; color: #f9d5d1 !important;
                border-color: rgba(248,113,113,.4) !important;
                background: #341b1e !important;
                font-size: 1rem;
            }
            .np-error-summary li { color: #f9d5d1 !important; }
            [data-testid="stNumberInput"] input,
            [data-testid="stTextInput"] input,
            [data-testid="stDateInput"] input,
            [data-testid="stNumberInput"] div[data-baseweb="input"] > div,
            [data-testid="stTextInput"] div[data-baseweb="input"] > div,
            [data-testid="stDateInput"] div[data-baseweb="input"] > div {
                color: #f0fdfa !important;
                -webkit-text-fill-color: #f0fdfa !important;
                background: #10272e !important;
                border-color: #45666b !important;
            }
            [data-testid="stNumberInput"] input::placeholder,
            [data-testid="stTextInput"] input::placeholder { color: #a8c0c4 !important; -webkit-text-fill-color: #a8c0c4 !important; }
            [data-testid="stNumberInput"] button,
            [data-testid="stDateInput"] button {
                color: #d7fbf4 !important; background: #14323a !important;
            }
            [data-testid="stNumberInput"] button svg,
            [data-testid="stDateInput"] button svg { fill: currentColor !important; }
            .stButton button[kind="primary"] p,
            .stFormSubmitButton button[kind="primaryFormSubmit"] p,
            .stButton button[data-testid="baseButton-primary"] p,
            .stFormSubmitButton button[data-testid="baseButton-primaryFormSubmit"] p { color: #002622 !important; }
            .stApp [data-testid="stMetricLabel"] p { color: #bdd3d5 !important; }
            .stApp [data-testid="stMetricValue"] { color: #f0fdfa !important; }
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] label { color: #d7e9e9 !important; }
            .np-review-term { color: #b7d0d1 !important; font-size: .9rem; }
            .np-review-value { font-size: .95rem; }
            .np-uacr-pill { font-size: .85rem; padding: .35rem .55rem; }
            @keyframes npStepReveal {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
            [class*="st-key-wizard_stage_"] { animation: npStepReveal 240ms ease-out both; }
            @media (max-width: 900px) {
                .st-key-hero_shell [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
                .st-key-hero_shell [data-testid="column"] { width: 100% !important; min-width: 0 !important; flex: 1 1 auto !important; }
                .np-logo-stage { width: min(60vw, 270px); }
            }
            @media (max-width: 700px) {
                [data-testid="stMainBlockContainer"] {
                    width: 100% !important;
                    padding: .75rem max(.75rem, env(safe-area-inset-right)) calc(2rem + env(safe-area-inset-bottom)) max(.75rem, env(safe-area-inset-left)) !important;
                }
                .st-key-hero_shell { padding: 1rem; border-radius: 20px; }
                .stApp [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: .65rem !important; }
                .stApp [data-testid="column"] { width: 100% !important; min-width: 0 !important; flex: 1 1 auto !important; }
                .np-logo-stage { width: min(42vw, 150px); margin: .1rem auto 0; padding: .65rem; }
                .np-hero-project-name { white-space: normal !important; overflow-wrap: anywhere; font-size: clamp(2.2rem, 10vw, 3.5rem) !important; }
                .np-hero-tagline { margin-top: .5rem; }
                .np-compact-header { gap: .65rem; padding: .6rem; }
                .np-compact-logo { width: 84px; height: 58px; flex-basis: 84px; }
                .np-compact-title { font-size: 1rem; }
                .np-section-title { font-size: clamp(1.7rem, 8vw, 2.4rem) !important; line-height: 1.12 !important; }
                [class*="st-key-question_card_"] [data-testid="stVerticalBlockBorderWrapper"],
                .st-key-review_card_shell [data-testid="stVerticalBlockBorderWrapper"] { padding: 1rem !important; }
                [class*="st-key-uacr_week_"] [data-testid="stNumberInput"] { grid-template-columns: 4.8rem minmax(0,1fr); }
                .np-review-grid { grid-template-columns: 1fr !important; }
                .risk-banner { min-height: 0; }
                .np-result-head { margin-top: .7rem; }
                [data-testid="stPlotlyChart"] { width: 100%; overflow: hidden; }
            }
            @media (max-width: 380px) {
                .np-hero-project-name { font-size: 2rem !important; }
                .np-progress-top { align-items: center; flex-direction: row; }
                .np-review-row { flex-direction: column; gap: .15rem; }
                .np-review-value { text-align: left; }
            }
            @media (prefers-reduced-motion: reduce) {
                [class*="st-key-wizard_stage_"] { animation: none !important; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def medical_lottie_animation() -> dict[str, Any]:
    """Return a small, self-contained medical Lottie composition."""
    teal = [0.369, 0.918, 0.831, 1.0]
    cyan = [0.133, 0.827, 0.933, 1.0]
    white = [0.925, 0.996, 0.988, 1.0]
    transform = {
        "ty": "tr",
        "p": {"a": 0, "k": [0, 0]},
        "a": {"a": 0, "k": [0, 0]},
        "s": {"a": 0, "k": [100, 100]},
        "r": {"a": 0, "k": 0},
        "o": {"a": 0, "k": 100},
        "sk": {"a": 0, "k": 0},
        "sa": {"a": 0, "k": 0},
    }
    return {
        "v": "5.10.0",
        "fr": 60,
        "ip": 0,
        "op": 180,
        "w": 320,
        "h": 280,
        "nm": "NephroPreempt clinical pulse",
        "ddd": 0,
        "assets": [],
        "layers": [
            {
                "ddd": 0,
                "ind": 1,
                "ty": 4,
                "nm": "Pulse ring",
                "sr": 1,
                "ks": {
                    "o": {
                        "a": 1,
                        "k": [
                            {"t": 0, "s": [64], "e": [0]},
                            {"t": 90, "s": [0], "e": [64]},
                            {"t": 180, "s": [64]},
                        ],
                    },
                    "r": {"a": 0, "k": 0},
                    "p": {"a": 0, "k": [160, 140, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {
                        "a": 1,
                        "k": [
                            {"t": 0, "s": [82, 82, 100], "e": [118, 118, 100]},
                            {"t": 90, "s": [118, 118, 100], "e": [82, 82, 100]},
                            {"t": 180, "s": [82, 82, 100]},
                        ],
                    },
                },
                "ao": 0,
                "shapes": [
                    {
                        "ty": "gr",
                        "nm": "Ring",
                        "it": [
                            {"d": 1, "ty": "el", "s": {"a": 0, "k": [174, 174]}, "p": {"a": 0, "k": [0, 0]}},
                            {"ty": "st", "c": {"a": 0, "k": cyan}, "o": {"a": 0, "k": 82}, "w": {"a": 0, "k": 3}, "lc": 2, "lj": 2},
                            transform,
                        ],
                    }
                ],
                "ip": 0,
                "op": 180,
                "st": 0,
                "bm": 0,
            },
            {
                "ddd": 0,
                "ind": 2,
                "ty": 4,
                "nm": "Orbit",
                "sr": 1,
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {"a": 1, "k": [{"t": 0, "s": [0], "e": [360]}, {"t": 180, "s": [360]}]},
                    "p": {"a": 0, "k": [160, 140, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {"a": 0, "k": [100, 100, 100]},
                },
                "ao": 0,
                "shapes": [
                    {
                        "ty": "gr",
                        "nm": "Dashed orbit",
                        "it": [
                            {"d": 1, "ty": "el", "s": {"a": 0, "k": [206, 206]}, "p": {"a": 0, "k": [0, 0]}},
                            {
                                "ty": "st",
                                "c": {"a": 0, "k": teal},
                                "o": {"a": 0, "k": 60},
                                "w": {"a": 0, "k": 2},
                                "lc": 2,
                                "lj": 2,
                                "d": [
                                    {"n": "d", "v": {"a": 0, "k": 10}},
                                    {"n": "g", "v": {"a": 0, "k": 13}},
                                ],
                            },
                            transform,
                        ],
                    }
                ],
                "ip": 0,
                "op": 180,
                "st": 0,
                "bm": 0,
            },
            {
                "ddd": 0,
                "ind": 3,
                "ty": 4,
                "nm": "Medical cross",
                "sr": 1,
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {"a": 0, "k": 0},
                    "p": {"a": 0, "k": [160, 140, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {
                        "a": 1,
                        "k": [
                            {"t": 0, "s": [96, 96, 100], "e": [104, 104, 100]},
                            {"t": 45, "s": [104, 104, 100], "e": [96, 96, 100]},
                            {"t": 90, "s": [96, 96, 100], "e": [104, 104, 100]},
                            {"t": 135, "s": [104, 104, 100], "e": [96, 96, 100]},
                            {"t": 180, "s": [96, 96, 100]},
                        ],
                    },
                },
                "ao": 0,
                "shapes": [
                    {
                        "ty": "gr",
                        "nm": "Cross shapes",
                        "it": [
                            {"d": 1, "ty": "rc", "s": {"a": 0, "k": [82, 24]}, "p": {"a": 0, "k": [0, 0]}, "r": {"a": 0, "k": 9}},
                            {"ty": "fl", "c": {"a": 0, "k": white}, "o": {"a": 0, "k": 100}, "r": 1},
                            transform,
                        ],
                    },
                    {
                        "ty": "gr",
                        "nm": "Cross vertical",
                        "it": [
                            {"d": 1, "ty": "rc", "s": {"a": 0, "k": [24, 82]}, "p": {"a": 0, "k": [0, 0]}, "r": {"a": 0, "k": 9}},
                            {"ty": "fl", "c": {"a": 0, "k": white}, "o": {"a": 0, "k": 100}, "r": 1},
                            transform,
                        ],
                    },
                ],
                "ip": 0,
                "op": 180,
                "st": 0,
                "bm": 0,
            },
        ],
        "markers": [],
    }


def render_medical_visual() -> None:
    if st_lottie is not None:
        with st.container(key="motion_visual"):
            st_lottie(
                medical_lottie_animation(),
                height=230,
                loop=True,
                speed=0.7,
                quality="high",
                key="nephropreempt_medical_lottie",
            )
        fallback_class = "np-lottie-fallback np-reduced-only"
    else:
        fallback_class = "np-lottie-fallback"

    st.markdown(
        f"""
        <div class="{fallback_class}" role="img" aria-label="Calm animated clinical pulse">
            <div class="np-vital-card">
                <svg aria-hidden="true" focusable="false" width="64" height="64" viewBox="0 0 64 64" fill="none">
                    <rect x="25" y="9" width="14" height="46" rx="6" fill="currentColor"/>
                    <rect x="9" y="25" width="46" height="14" rx="6" fill="currentColor"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(expanded: bool = True) -> None:
    if not expanded:
        st.markdown(
            f"""
            <header class="np-compact-header">
                <img class="np-compact-logo" src="{PROJECT_LOGO_DATA_URI}" alt="NephroPreempt logo">
                <div class="np-compact-title">{APP_TITLE}</div>
            </header>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.container(key="hero_shell"):
        copy_col, visual_col = st.columns([1.2, 0.8], gap="large", vertical_alignment="center")
        with copy_col:
            st.markdown(
                f"""
                <div class="np-eyebrow"><span class="np-eyebrow-dot"></span>Kidney risk assessment</div>
                <h1 class="np-hero-project-name">{APP_TITLE}</h1>
                <div class="np-hero-tagline">See risk earlier. Act with clarity.</div>
                """,
                unsafe_allow_html=True,
            )
        with visual_col:
            st.markdown(
                f"""
                <div class="np-logo-stage">
                    <img class="np-project-logo" src="{PROJECT_LOGO_DATA_URI}" alt="NephroPreempt project logo">
                </div>
                """,
                unsafe_allow_html=True,
            )


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


WIZARD_STEPS = (
    ("Welcome", "Patient name"),
    ("Profile", "Date of birth"),
    ("Profile", "Height and weight"),
    ("Profile", "Biological sex"),
    ("Vitals", "Systolic blood pressure"),
    ("History", "Diabetes"),
    ("History", "Hypertension"),
    ("History", "Smoking status"),
    ("Treatment", "ACE inhibitor or ARB use"),
    ("Laboratory", "12-week UACR series"),
    ("Review", "Verify and analyze"),
)

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
        "wizard_schema_version": 2,
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
    # Reassignment keeps answers durable across the one-question screens.
    for key in WIZARD_DATA_KEYS:
        st.session_state[key] = st.session_state[key]

    try:
        current_step = int(st.session_state["wizard_step"])
    except (TypeError, ValueError):
        current_step = 1
    st.session_state["wizard_step"] = int(np.clip(current_step, 1, len(WIZARD_STEPS)))
    st.session_state["wizard_schema_version"] = 2


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
        f'<div class="np-error-summary" role="alert"><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def wizard_phase_index(step: int) -> int:
    if step <= 4:
        return 0
    if step <= 8:
        return 1
    if step == 9:
        return 2
    if step == 10:
        return 3
    return 4


def render_wizard_progress(step: int, complete: bool = False) -> None:
    safe_step = int(np.clip(step, 1, len(WIZARD_STEPS)))
    percentage = 100 if complete else int(round((safe_step - 1) / (len(WIZARD_STEPS) - 1) * 100))
    status_text = "Assessment complete" if complete else WIZARD_STEPS[safe_step - 1][1]
    st.markdown(
        f"""
        <div class="np-progress-card">
            <div class="np-progress-top">
                <span class="np-progress-label">{html.escape(status_text)}</span>
                <span class="np-progress-value">{safe_step} / {len(WIZARD_STEPS)}</span>
            </div>
            <div class="np-progress-track" role="progressbar" aria-label="Assessment progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percentage}">
                <div class="np-progress-fill" style="width:{percentage}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_heading(kicker: str, title: str, copy: str = "") -> None:
    st.markdown(
        f'<div class="np-section-heading"><h2 class="np-section-title">{html.escape(title)}</h2></div>',
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


def go_forward(next_step: int, validator: Any | None = None) -> None:
    errors = list(validator() if validator is not None else [])
    if errors:
        st.session_state["step_errors"] = errors
        st.rerun()
    invalidate_analysis()
    move_to_step(next_step)


def render_navigation(
    *,
    step: int,
    next_label: str,
    validator: Any | None = None,
    back_label: str | None = None,
) -> None:
    if step == 1:
        if st.button(next_label, type="primary", key=f"wizard_next_{step}", width="stretch"):
            go_forward(step + 1, validator)
        return

    back_col, next_col = st.columns(2)
    with back_col:
        if st.button(back_label or "Back", key=f"wizard_back_{step}", width="stretch"):
            move_to_step(step - 1)
    with next_col:
        if st.button(next_label, type="primary", key=f"wizard_next_{step}", width="stretch"):
            go_forward(step + 1, validator)


def render_name_step() -> None:
    render_step_heading(
        "Welcome / Question 01",
        "What should we call you?",
        "Start with the patient name. It is used only to personalize the results greeting and never enters the prediction model.",
    )
    render_error_summary(list(st.session_state.get("step_errors", [])))
    with st.container(border=True, key="question_card_name"):

        st.text_input(
            "Patient name",
            key="patient_name",
            placeholder="Enter the patient's name",
            max_chars=80,
            on_change=invalidate_analysis,
        )
    render_navigation(step=1, next_label="Continue to date of birth", validator=patient_name_errors)


def render_dob_step() -> None:
    render_step_heading(
        "Profile / Question 02",
        "When was the patient born?",
        "Choose the date directly from the calendar. Reference matching happens securely in the background.",
    )
    render_error_summary(list(st.session_state.get("step_errors", [])))
    with st.container(border=True, key="question_card_dob"):

        st.date_input(
            "Date of birth",
            min_value=dob_for_age(AGE_MAX),
            max_value=dob_for_age(AGE_MIN),
            format="YYYY-MM-DD",
            key="patient_dob",
            on_change=invalidate_analysis,
        )
    render_navigation(
        step=2,
        back_label="Back to name",
        next_label="Continue to measurements",
        validator=dob_errors,
    )


def render_measurements_step() -> None:
    render_step_heading(
        "Profile / Question 03",
        "Add height and weight",
        "Enter the two body measurements together. Derived model values stay hidden during the assessment.",
    )
    render_error_summary(list(st.session_state.get("step_errors", [])))
    with st.container(border=True, key="question_card_measurements"):

        height_col, weight_col = st.columns(2, gap="large")
        with height_col:
            st.number_input(
                "Height (cm)",
                min_value=HEIGHT_MIN,
                max_value=HEIGHT_MAX,
                step=0.1,
                format="%.1f",
                key="patient_height_cm",
                on_change=invalidate_analysis,
            )
        with weight_col:
            st.number_input(
                "Weight (kg)",
                min_value=WEIGHT_MIN,
                max_value=WEIGHT_MAX,
                step=0.1,
                format="%.1f",
                key="patient_weight_kg",
                on_change=invalidate_analysis,
            )
    render_navigation(
        step=3,
        back_label="Back to date of birth",
        next_label="Continue to biological sex",
        validator=measurement_errors,
    )


def render_sex_step() -> None:
    render_step_heading(
        "Profile / Question 04",
        "Select biological sex at birth",
        "This value supports the bundled model encoding and selects the appropriate UACR reference curve.",
    )
    render_error_summary(list(st.session_state.get("step_errors", [])))
    with st.container(border=True, key="question_card_sex"):

        st.radio(
            "Biological sex at birth",
            ["Male", "Female"],
            index=None,
            horizontal=True,
            key="patient_sex",
            on_change=invalidate_analysis,
        )
    render_navigation(
        step=4,
        back_label="Back to measurements",
        next_label="Continue to blood pressure",
        validator=lambda: choice_errors("patient_sex", "a biological sex", ("Male", "Female")),
    )


def render_sbp_step() -> None:
    render_step_heading(
        "Vitals / Question 05",
        "What is the baseline systolic pressure?",
        "Use the upper blood-pressure number from a representative baseline measurement.",
    )
    render_error_summary(list(st.session_state.get("step_errors", [])))
    with st.container(border=True, key="question_card_sbp"):

        st.slider(
            "Systolic blood pressure (mmHg)",
            min_value=SBP_MIN,
            max_value=SBP_MAX,
            step=1,
            key="baseline_sbp",
            on_change=invalidate_analysis,
        )
    render_navigation(
        step=5,
        back_label="Back to biological sex",
        next_label="Continue to diabetes",
    )


def render_binary_step(
    *,
    step: int,
    state_key: str,
    kicker: str,
    title: str,
    copy: str,
    field_label: str,
    help_text: str,
    back_label: str,
    next_label: str,
) -> None:
    render_step_heading(kicker, title, copy)
    render_error_summary(list(st.session_state.get("step_errors", [])))
    with st.container(border=True, key=f"question_card_{state_key}"):
        st.radio(
            field_label,
            ["Yes", "No"],
            index=None,
            horizontal=True,
            key=state_key,
            on_change=invalidate_analysis,
        )
    render_navigation(
        step=step,
        back_label=back_label,
        next_label=next_label,
        validator=lambda: choice_errors(state_key, f"Yes or No for {field_label.lower()}"),
    )


def render_diabetes_step() -> None:
    render_binary_step(
        step=6,
        state_key="has_diabetes",
        kicker="History / Question 06",
        title="Is diabetes documented?",
        copy="Answer from the current medical record.",
        field_label="Diagnosed diabetes",
        help_text="Choose Yes when diabetes is documented in the current medical record.",
        back_label="Back to blood pressure",
        next_label="Continue to hypertension",
    )


def render_hypertension_step() -> None:
    render_binary_step(
        step=7,
        state_key="has_hypertension",
        kicker="History / Question 07",
        title="Is hypertension documented?",
        copy="Keep this separate from the baseline pressure value entered earlier.",
        field_label="Diagnosed hypertension",
        help_text="Choose Yes when hypertension is documented in the current medical record.",
        back_label="Back to diabetes",
        next_label="Continue to smoking status",
    )


def render_smoking_step() -> None:
    render_binary_step(
        step=8,
        state_key="is_smoker",
        kicker="History / Question 08",
        title="Does the patient currently smoke?",
        copy="The bundled model uses current active smoking status.",
        field_label="Current active smoker",
        help_text="Choose Yes for current active smoking. Former smoking is not represented in the model.",
        back_label="Back to hypertension",
        next_label="Continue to treatment",
    )


def render_medication_step() -> None:
    render_binary_step(
        step=9,
        state_key="uses_acei_arb",
        kicker="Treatment / Question 09",
        title="Is an ACE inhibitor or ARB in use?",
        copy="Confirm current kidney-protective medication use from the medication list.",
        field_label="Current ACE inhibitor / ARB use",
        help_text="Choose Yes for current use of an ACE inhibitor or angiotensin II receptor blocker.",
        back_label="Back to smoking status",
        next_label="Continue to UACR readings",
    )


def out_of_distribution_weeks(values: list[float]) -> list[int]:
    upper_bounds = resize_series(LSTM_UACR_MAX, len(values))
    return [index for index, (value, upper) in enumerate(zip(values, upper_bounds), start=1) if value > float(upper)]


def render_uacr_step() -> None:
    render_step_heading(
        "Laboratory / Question 10",
        "Enter the 12-week UACR series",
        "Add one urine albumin-to-creatinine ratio result per consecutive week so the trend model can read the full trajectory.",
    )
    render_error_summary(list(st.session_state.get("step_errors", [])))
    with st.container(border=True, key="question_card_uacr"):

        st.markdown(
            '<div class="np-inline-note">UACR in mg/g · Week 1 is oldest; Week 12 is newest.</div>',
            unsafe_allow_html=True,
        )
        with st.form("uacr_series_form", clear_on_submit=False, border=False):
            for first_week in range(1, 13, 3):
                week_columns = st.columns(3, gap="medium")
                for column, index in zip(week_columns, range(first_week, first_week + 3)):
                    with column:
                        st.number_input(
                            f"Week {index}",
                            min_value=UACR_MIN,
                            max_value=UACR_MAX,
                            step=0.1,
                            format="%.1f",
                            key=f"uacr_week_{index}",
                        )
            back_col, next_col = st.columns(2)
            with back_col:
                back_clicked = st.form_submit_button(
                    "Back to treatment",
                    type="secondary",
                    width="stretch",
                )
            with next_col:
                next_clicked = st.form_submit_button(
                    "Continue to review",
                    type="primary",
                    width="stretch",
                )

    if back_clicked:
        invalidate_analysis()
        move_to_step(9)
    if next_clicked:
        invalidate_analysis()
        errors = all_wizard_errors()
        if errors:
            st.session_state["step_errors"] = errors
            st.rerun()
        move_to_step(11)

    outlier_weeks = out_of_distribution_weeks(uacr_values_from_state())
    if outlier_weeks:
        st.warning(
            "One or more values are above the corresponding weekly maximum observed in the model's training data "
            f"(week{'s' if len(outlier_weeks) != 1 else ''} {', '.join(map(str, outlier_weeks))}). "
            "You can continue, but the result needs additional clinical caution."
        )


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
        f'<span class="np-uacr-pill">W{index}: {value:.1f}</span>'
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


def render_review_step(bundle: AssetBundle) -> None:
    render_step_heading(
        "Review / Question 11",
        "Everything ready?",
        "Check the full assessment before running the models. Only the values you provided appear below.",
    )
    review_errors = all_wizard_errors()
    render_error_summary(review_errors or list(st.session_state.get("step_errors", [])))
    if st.session_state.get("analysis_error"):
        st.error(str(st.session_state["analysis_error"]))

    inputs = inputs_from_state()
    uacr_values = uacr_values_from_state()
    with st.container(border=True, key="review_card_shell"):
        render_review_cards(inputs, uacr_values)

        st.button(
            "Edit assessment",
            key="edit_assessment",
            width="stretch",
            on_click=prepare_step_change,
            args=(1,),
        )

        outlier_weeks = out_of_distribution_weeks(uacr_values)
        if outlier_weeks:
            st.warning(
                "Training-range caution: readings for "
                f"week{'s' if len(outlier_weeks) != 1 else ''} {', '.join(map(str, outlier_weeks))} "
                "are above values observed during LSTM training."
            )

        st.checkbox(
            "I have reviewed these entries and understand that the result supports, but does not replace, clinical judgment.",
            key="review_confirmed",
        )

        back_col, submit_col = st.columns(2)
        with back_col:
            st.button(
                "Back to UACR",
                key="wizard_back_review",
                width="stretch",
                on_click=prepare_step_change,
                args=(10,),
            )
        with submit_col:
            analyze_clicked = st.button(
                "Run kidney risk analysis",
                type="primary",
                key="submit_assessment",
                width="stretch",
                disabled=bool(bundle.errors)
                or bool(review_errors)
                or not bool(st.session_state["review_confirmed"]),
            )

    if bundle.errors:
        st.error("Analysis is unavailable until the clinical models load successfully.")

    if analyze_clicked:
        errors = all_wizard_errors()
        if errors:
            st.session_state["step_errors"] = errors
            st.rerun()
        try:
            with st.spinner("Running static and trend models…", show_time=True):
                result = run_prediction(bundle, inputs, uacr_values)
        except Exception as exc:
            st.session_state["analysis_error"] = str(exc)
            st.rerun()
        st.session_state["analysis_result"] = result
        st.session_state["submitted_snapshot"] = state_snapshot(inputs, uacr_values)
        st.session_state["analysis_error"] = ""
        st.session_state["step_errors"] = []
        st.rerun()


def render_system_status(bundle: AssetBundle) -> None:
    if bundle.errors:
        st.error("The clinical models could not be loaded. Analysis is unavailable.")
    for warning in bundle.warnings:
        st.warning(warning)


def render_system_diagnostics(bundle: AssetBundle) -> None:
    if not bundle.errors:
        return
    with st.expander("Technical details", expanded=False):
        for error in bundle.errors:
            st.write(error)
        if st.button("Retry loading models", key="reload_clinical_assets"):
            load_assets.clear()
            st.session_state.pop("clinical_asset_bundle", None)
            st.rerun()


def render_chart(result: PredictionResult) -> None:
    weeks = np.arange(1, WEEKS_IN_SERIES + 1)
    raw = result.uacr_processing.raw
    smooth = result.uacr_processing.smoothed
    reference = result.reference.curve
    color = RISK_BANDS[result.risk_level].get("dark_color", RISK_BANDS[result.risk_level]["color"])

    if go is None:
        chart_frame = pd.DataFrame(
            {
                "Week": weeks,
                "Raw UACR": raw,
                "Smoothed UACR": smooth,
                                "Reference": reference,
            }
        )
        st.dataframe(chart_frame, width="stretch", hide_index=True)
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=smooth,
            mode="lines+markers",
            name="UACR trend",
            line=dict(color=color, width=4),
            marker=dict(size=8, color=color, line=dict(color="#ffffff", width=2)),
            fill="tozeroy",
            fillcolor="rgba(8, 145, 178, 0.07)",
            hovertemplate="Week %{x}<br>Smoothed UACR: %{y:.1f} mg/g<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=raw,
            mode="lines+markers",
            name="Readings",
            line=dict(color="#94a3b8", width=2, dash="dot"),
            marker=dict(size=8, color="#94a3b8", line=dict(color="#07161b", width=1), symbol="circle"),
            hovertemplate="Week %{x}<br>Raw UACR: %{y:.1f} mg/g<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=reference,
            mode="lines",
            name="Healthy reference",
            line=dict(color="#22d3ee", width=3, dash="dash"),
            hovertemplate="Week %{x}<br>Healthy P50: %{y:.1f} mg/g<extra></extra>",
        )
    )
    fig.update_layout(
        height=370,
        template="plotly_dark",
        paper_bgcolor="#061419",
        plot_bgcolor="#081a20",
        margin=dict(l=12, r=12, t=60, b=12),
        xaxis_title="Week",
        yaxis_title="UACR (mg/g)",
        hovermode="x unified",
        font=dict(family="Noto Sans, Segoe UI, sans-serif", color="#c9dcdf", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=11)),
        uirevision="nephropreempt-uacr",
    )
    fig.update_xaxes(dtick=1, gridcolor="#17333a", zerolinecolor="#17333a", fixedrange=False)
    fig.update_yaxes(gridcolor="#17333a", zerolinecolor="#17333a", rangemode="tozero", fixedrange=False)
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "png", "scale": 2}},
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


def render_result(
    result: PredictionResult,
    inputs: StaticInputs,
    uacr_values: list[float],
    patient_name: str,
    generated_at: datetime | None = None,
) -> None:
    theme = RISK_BANDS[result.risk_level]
    color = theme.get("dark_color", theme["color"])
    surface = theme["surface"]
    generated_at = generated_at or datetime.now()
    safe_patient_name = html.escape(patient_name.strip() or "Patient")

    st.markdown(
        f"""
        <div class="np-result-head">
            <div>
                <h2>Kidney risk assessment</h2>
                <p class="np-result-greeting">{safe_patient_name}, your result is ready for clinical review.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="risk-banner" style="background:{surface}; color:{color};">
                <div class="risk-kicker">Estimated kidney risk</div>
                <div class="risk-score" style="color:{color};">{result.final_percent:.1f}%</div>
                <div class="risk-level" style="color:{color};">{html.escape(result.risk_level)}</div>
                <div class="directive">{html.escape(theme["directive"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pdf_bytes = build_simple_pdf_report(result, inputs, uacr_values)
        st.download_button(
            "Download clinical PDF report",
            data=pdf_bytes,
            file_name=f"NephroPreempt_Report_{generated_at.strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            width="stretch",
        )

    with right:
        static_col, trend_col = st.columns(2)
        static_col.metric("Clinical profile", f"{result.p_xgb * 100.0:.1f}%")
        trend_col.metric("12-week trend", f"{result.p_trend * 100.0:.1f}%")
        st.metric("Latest UACR (mg/g)", f"{uacr_values[-1]:.1f}")
        if result.severity_floor_applied:
            st.warning(result.severity_floor_label)

    st.markdown("### 12-week UACR trend")
    direction = "rising" if result.uacr_processing.raw_slope > 0.05 else "falling" if result.uacr_processing.raw_slope < -0.05 else "stable"
    st.markdown(f"UACR readings are **{direction}** overall.")
    render_chart(result)

    with st.expander("View weekly UACR readings"):
        st.dataframe(
            pd.DataFrame({"Week": np.arange(1, WEEKS_IN_SERIES + 1), "UACR (mg/g)": result.uacr_processing.raw}),
            width="stretch",
            hide_index=True,
        )

    st.info(
        "Clinical decision support only. Interpret this result with the full patient record and local care protocols. "
        "It is not a diagnosis or an emergency triage tool."
    )
    if st.button("Edit assessment", key="edit_completed_assessment"):
        st.session_state["analysis_result"] = None
        st.session_state["submitted_snapshot"] = None
        st.session_state["review_confirmed"] = False
        st.session_state["wizard_step"] = 1
        st.rerun()


def main() -> None:
    configure_page()
    configure_dark_experience()
    initialize_session_state()
    result = st.session_state.get("analysis_result")
    snapshot = st.session_state.get("submitted_snapshot")
    step = int(st.session_state["wizard_step"])
    render_header(expanded=result is None and step == 1)
    if "clinical_asset_bundle" not in st.session_state:
        with st.spinner("Verifying clinical assets…"):
            st.session_state["clinical_asset_bundle"] = load_assets()
    bundle = st.session_state["clinical_asset_bundle"]
    render_system_status(bundle)

    if result is not None and isinstance(snapshot, dict):
        render_wizard_progress(len(WIZARD_STEPS), complete=True)
        with st.container(key="result_panel"):
            render_result(
                result,
                snapshot["inputs"],
                snapshot["uacr_values"],
                patient_name=str(snapshot.get("patient_name", "Patient")),
                generated_at=snapshot.get("created_at"),
            )
    else:
        render_wizard_progress(step)
        with st.container(key=f"wizard_stage_{step}"):
            if step == 1:
                render_name_step()
            elif step == 2:
                render_dob_step()
            elif step == 3:
                render_measurements_step()
            elif step == 4:
                render_sex_step()
            elif step == 5:
                render_sbp_step()
            elif step == 6:
                render_diabetes_step()
            elif step == 7:
                render_hypertension_step()
            elif step == 8:
                render_smoking_step()
            elif step == 9:
                render_medication_step()
            elif step == 10:
                render_uacr_step()
            elif step == 11:
                render_review_step(bundle)

    render_system_diagnostics(bundle)


if __name__ == "__main__":
    main()
