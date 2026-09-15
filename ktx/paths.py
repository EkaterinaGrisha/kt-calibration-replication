"""Пути внутри репозитория воспроизведения.

Раскладка плоская: ktx/, scripts/, tests/, artifacts/, paper/. Всё разрешается
относительно корня репозитория, поэтому сценарии не зависят от текущего каталога.

PYKT_ROOT указывает на каталог с последовательностями библиотеки предобработки.
В этом репозитории его нет: сценарии, которым он нужен, перечислены в README и
приведены ради проверяемости, а их результаты лежат готовыми таблицами.
"""
from __future__ import annotations

from pathlib import Path

# ktx/paths.py -> ktx -> корень репозитория
REPO_ROOT = Path(__file__).resolve().parents[1]

RESEARCH_DIR = REPO_ROOT
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PYKT_ROOT = REPO_ROOT / "vendor" / "pykt-toolkit"
