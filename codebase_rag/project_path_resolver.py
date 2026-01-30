from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from . import logs as ls

if TYPE_CHECKING:
    from .config import AppConfig


class ProjectPathResolver:
    def __init__(self, mappings: dict[str, str] | None = None):
        self._mappings: dict[str, Path] = {}

        if mappings:
            for name, path in mappings.items():
                self._mappings[name] = Path(path).resolve()
            logger.info(
                ls.RESOLVER_INIT_MAPPED.format(
                    count=len(mappings), projects=list(mappings.keys())
                )
            )
        else:
            from .config import settings

            default_path = Path(settings.TARGET_REPO_PATH).resolve()
            default_project = default_path.name
            self._mappings[default_project] = default_path
            logger.info(
                ls.RESOLVER_INIT_DEFAULT.format(
                    project=default_project, path=default_path
                )
            )

    def extract_project_name(self, qualified_name: str) -> str:
        sorted_projects = sorted(self._mappings.keys(), key=len, reverse=True)

        for project_name in sorted_projects:
            if qualified_name.startswith(f"{project_name}."):
                logger.debug(
                    ls.RESOLVER_EXTRACT_SUCCESS.format(
                        fqn=qualified_name, project=project_name
                    )
                )
                return str(project_name)

        fallback = qualified_name.split(".")[0]
        logger.warning(
            ls.RESOLVER_EXTRACT_FALLBACK.format(fqn=qualified_name, fallback=fallback)
        )
        return fallback

    def get_project_path(self, project_name: str) -> Path:
        if project_name not in self._mappings:
            available = ", ".join(self._mappings.keys())
            raise KeyError(
                ls.RESOLVER_PROJECT_NOT_FOUND.format(
                    project=project_name, available=available
                )
            )

        return self._mappings[project_name]

    def resolve_path_from_fqn(self, qualified_name: str) -> Path:
        project_name = self.extract_project_name(qualified_name)
        return self.get_project_path(project_name)

    def list_projects(self) -> list[str]:
        return list(self._mappings.keys())

    def add_project(self, name: str, path: str) -> None:
        self._mappings[name] = Path(path).resolve()
        logger.info(ls.RESOLVER_PROJECT_ADDED.format(name=name, path=path))

    def remove_project(self, name: str) -> None:
        if name not in self._mappings:
            raise KeyError(
                ls.RESOLVER_PROJECT_NOT_FOUND.format(
                    project=name, available=", ".join(self._mappings.keys())
                )
            )
        del self._mappings[name]
        logger.info(ls.RESOLVER_PROJECT_REMOVED.format(name=name))

    @classmethod
    def from_config(cls, config: AppConfig) -> ProjectPathResolver:
        mappings = config.get_project_mappings()
        return cls(mappings)
