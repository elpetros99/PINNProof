from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys


@dataclass(frozen=True)
class _VendoredPackage:
    public_package: str
    upstream_root: Path

    def matches(self, fullname: str) -> bool:
        if fullname == self.public_package or fullname.startswith(f"{self.public_package}."):
            return False

        first_segment = fullname.split(".", 1)[0]
        return _has_upstream_child(self.upstream_root, first_segment)

    def target_name_for(self, fullname: str) -> str:
        return f"{self.public_package}.{fullname}"


class _VendoredAliasLoader(importlib.abc.Loader):
    def __init__(self, alias_name: str, target_name: str) -> None:
        self._alias_name = alias_name
        self._target_name = target_name

    def create_module(self, spec):
        module = importlib.import_module(self._target_name)
        sys.modules[self._alias_name] = module
        return module

    def exec_module(self, module) -> None:
        return None


class _VendoredAliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._packages: dict[str, _VendoredPackage] = {}

    def register(self, public_package: str, upstream_root: Path) -> None:
        self._packages[public_package] = _VendoredPackage(
            public_package=public_package,
            upstream_root=upstream_root.resolve(),
        )

    def find_spec(self, fullname: str, path=None, target=None):
        matches: list[tuple[_VendoredPackage, str, object]] = []

        for package in self._packages.values():
            if not package.matches(fullname):
                continue

            target_name = package.target_name_for(fullname)
            target_spec = importlib.util.find_spec(target_name)
            if target_spec is None:
                continue

            matches.append((package, target_name, target_spec))

        if not matches:
            return None

        if len(matches) > 1:
            providers = ", ".join(sorted(match[0].public_package for match in matches))
            raise ImportError(
                f"Ambiguous vendored import '{fullname}'. It is exposed by multiple packages: {providers}."
            )

        _, target_name, target_spec = matches[0]
        loader = _VendoredAliasLoader(alias_name=fullname, target_name=target_name)
        is_package = target_spec.submodule_search_locations is not None
        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=target_spec.origin,
            is_package=is_package,
        )


def _has_upstream_child(upstream_root: Path, name: str) -> bool:
    return (upstream_root / name).is_dir() or (upstream_root / f"{name}.py").is_file()


_ALIAS_FINDER = _VendoredAliasFinder()


def configure_vendored_package(public_package: str, upstream_root: Path) -> list[str]:
    upstream_root = upstream_root.resolve()
    if not upstream_root.is_dir() or not any(upstream_root.iterdir()):
        raise ModuleNotFoundError(
            f"Vendored package '{public_package}' is missing at '{upstream_root}'. "
            "Initialize submodules with 'git submodule update --init --recursive'."
        )

    if _ALIAS_FINDER not in sys.meta_path:
        sys.meta_path.append(_ALIAS_FINDER)

    _ALIAS_FINDER.register(public_package=public_package, upstream_root=upstream_root)
    return [str(upstream_root)]
