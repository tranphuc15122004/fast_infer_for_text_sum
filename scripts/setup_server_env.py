#!/usr/bin/env python3
"""Khởi tạo và kiểm tra runtime dùng chung trên server benchmark.

Server production không tạo virtualenv riêng cho project này. Script dùng
``python3`` hệ thống, mặc định yêu cầu Python 3.12, và đặt dữ liệu ổn định
ngoài repository. Script không tải package/model và không ghi đè master config
đã có.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SHARED_DATA_DIR = Path(
    "/workspace/storage-shared/nlp/dungdx4/phuc_projects/data"
)
DATASETS = ("longbench_200", "representative_100")


class SetupError(RuntimeError):
    """Lỗi setup có thông báo có thể xử lý bởi operator."""


def _resolve(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _shell_value(value: Path | str) -> str:
    return shlex.quote(str(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--init",
        action="store_true",
        help="tạo thư mục/config/pointer/link còn thiếu, không chạy preflight",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="chỉ kiểm tra, không tạo hoặc sửa file nào",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="init rồi chạy toàn bộ preflight (mặc định)",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="root repository (mặc định: vị trí script)",
    )
    parser.add_argument(
        "--shared-data-dir",
        default=str(DEFAULT_SHARED_DATA_DIR),
        help=f"thư mục data/config ổn định (mặc định: {DEFAULT_SHARED_DATA_DIR})",
    )
    parser.add_argument(
        "--master-config",
        default=None,
        help="master config override; mặc định <shared-data-dir>/fast_infer_master.env",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="interpreter cần kiểm tra trên server (mặc định: python3)",
    )
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="bỏ qua check_shared_env.py; chỉ dùng khi debug filesystem",
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="bỏ qua validator LongBench; dùng trước khi copy dataset vào shared dir",
    )
    return parser


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    repo = _resolve(args.repo_dir)
    shared = _resolve(args.shared_data_dir)
    master = _resolve(args.master_config) if args.master_config else shared / "fast_infer_master.env"
    return {
        "repo": repo,
        "shared": shared,
        "master": master,
        "pointer": repo / "config" / "master.path",
        "example": repo / "docs" / "fast_infer_master.example.env",
        "validator": repo / "scripts" / "validate_longbench_200.py",
        "shared_longbench": shared / "longbench_200",
        "shared_representative": shared / "representative_100",
    }


def _generated_master_block(paths: dict[str, Path]) -> str:
    # Chỉ ghi các biến thuộc trách nhiệm của bootstrap. Model/checkpoint paths
    # vẫn do operator khai báo trong master config.
    return (
        "\n\n# BEGIN setup_server_env.py managed defaults\n"
        f"FI_PYTHON={_shell_value('python3')}\n"
        "FI_DEVICE=cuda\n"
        "FI_OFFLINE=1\n"
        f"DATA_ROOT={_shell_value(paths['shared'])}\n"
        f"LONG_BENCH_DATA_DIR={_shell_value(paths['shared_longbench'])}\n"
        f"LONG_BENCH_OUTPUT_DIR={_shell_value(paths['repo'] / 'outputs' / 'longbench_200')}\n"
        "LONG_BENCH_LOCAL_FILES_ONLY=1\n"
        "# END setup_server_env.py managed defaults\n"
    )


def _init_master(paths: dict[str, Path]) -> None:
    master = paths["master"]
    master.parent.mkdir(parents=True, exist_ok=True)
    if master.exists():
        print(f"KEEP master config: {master}")
        return
    example = paths["example"]
    if not example.is_file():
        raise SetupError(f"không tìm thấy master example: {example}")
    text = example.read_text(encoding="utf-8")
    master.write_text(text + _generated_master_block(paths), encoding="utf-8")
    print(f"CREATE master config: {master}")


def _init_pointer(paths: dict[str, Path]) -> None:
    pointer = paths["pointer"]
    pointer.parent.mkdir(parents=True, exist_ok=True)
    expected = str(paths["master"]) + "\n"
    current = pointer.read_text(encoding="utf-8") if pointer.exists() else ""
    if current == expected:
        print(f"KEEP master pointer: {pointer}")
        return
    pointer.write_text(
        "# Đường dẫn master config dùng chung do setup_server_env.py quản lý.\n"
        + expected,
        encoding="utf-8",
    )
    print(f"UPDATE master pointer: {pointer} -> {paths['master']}")


def _init_data_dirs(paths: dict[str, Path]) -> None:
    paths["shared"].mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        path = paths["shared"] / name
        if path.exists():
            print(f"KEEP shared data directory: {path}")
        else:
            path.mkdir()
            print(f"CREATE shared data directory: {path}")


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _init_link(repo: Path, shared: Path, name: str) -> None:
    link = repo / "data" / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(link):
        link.symlink_to(shared, target_is_directory=True)
        print(f"LINK data/{name}: {link} -> {shared}")
        return
    if link.is_symlink() and _same_path(link, shared):
        print(f"KEEP data link: {link} -> {shared}")
        return
    if link.is_dir():
        # Không xoá dữ liệu đã checkout. Runner LongBench dùng đường dẫn
        # absolute trong master config; symlink chỉ là tiện ích tương thích.
        print(f"KEEP existing repository directory: {link}")
        return
    raise SetupError(
        f"data target đã tồn tại nhưng không phải directory/symlink: {link}; "
        "không tự động ghi đè"
    )


def initialize(paths: dict[str, Path]) -> None:
    _init_data_dirs(paths)
    _init_master(paths)
    _init_pointer(paths)
    _init_link(paths["repo"], paths["shared_longbench"], "longbench_200")
    _init_link(paths["repo"], paths["shared_representative"], "representative_100")


def _python_executable(value: str) -> str:
    candidate = shutil.which(value) if "/" not in value else value
    if not candidate or not os.access(candidate, os.X_OK):
        raise SetupError(f"không tìm thấy executable Python: {value}")
    return candidate


def _check_python(value: str) -> str:
    executable = _python_executable(value)
    probe = subprocess.run(
        [
            executable,
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}'); "
            "sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)",
        ],
        text=True,
        capture_output=True,
    )
    version = probe.stdout.strip() or "unknown"
    if probe.returncode != 0:
        raise SetupError(
            f"Python phải là 3.12, nhưng {executable} trả về {version}. "
            "Hãy kiểm tra PATH/python3 trên server."
        )
    print(f"PASS Python: {executable} ({version})")
    return executable


def _pointer_target(pointer: Path) -> Path | None:
    if not pointer.is_file():
        return None
    for line in pointer.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            return _resolve(value)
    return None


def _check_master(paths: dict[str, Path]) -> None:
    master = paths["master"]
    if not master.is_file():
        raise SetupError(f"master config chưa tồn tại: {master}; chạy --init trước")
    if _pointer_target(paths["pointer"]) != master:
        raise SetupError(
            f"master pointer không trỏ tới {master}: {paths['pointer']}; chạy --init để cập nhật"
        )
    syntax = subprocess.run(["bash", "-n", str(master)], text=True, capture_output=True)
    if syntax.returncode != 0:
        raise SetupError(f"master config sai Bash syntax: {syntax.stderr.strip()}")
    print(f"PASS master config: {master}")


def _check_data(paths: dict[str, Path], *, validate: bool, python: str) -> None:
    if not paths["shared"].is_dir():
        raise SetupError(f"shared data directory chưa tồn tại: {paths['shared']}")
    for name in DATASETS:
        path = paths["shared"] / name
        if not path.is_dir():
            raise SetupError(f"dataset directory chưa tồn tại: {path}")
        print(f"PASS dataset directory: {path}")

    if not validate:
        print("SKIP LongBench validator: --skip-data-validation")
        return
    validator = paths["validator"]
    if not validator.is_file():
        raise SetupError(f"không tìm thấy validator: {validator}")
    command = [
        python,
        str(validator),
        "--data-dir",
        str(paths["shared_longbench"]),
        "--expected-count",
        "200",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        raise SetupError(
            "LongBench validation thất bại; hãy kiểm tra data/longbench_200.\n"
            + result.stderr.strip()
        )
    print("PASS LongBench canonical data: 5 dataset files, 200 records/dataset")


def _check_dependencies(paths: dict[str, Path], python: str, *, skip: bool) -> None:
    if skip:
        print("SKIP dependency preflight: --skip-dependencies")
        return
    checker = paths["repo"] / "scripts" / "check_shared_env.py"
    if not checker.is_file():
        raise SetupError(f"không tìm thấy dependency checker: {checker}")
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    result = subprocess.run([python, str(checker)], env=env, text=True)
    if result.returncode != 0:
        raise SetupError(
            "dependency preflight thất bại; sửa package/CUDA stack trước khi benchmark"
        )
    print("PASS shared dependency preflight")


def check(paths: dict[str, Path], args: argparse.Namespace) -> None:
    python = _check_python(args.python)
    _check_master(paths)
    _check_data(
        paths,
        validate=not args.skip_data_validation,
        python=python,
    )
    _check_dependencies(paths, python, skip=args.skip_dependencies)
    print("\nServer environment preflight: PASS")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _paths(args)
    try:
        if args.check:
            check(paths, args)
        elif args.init:
            initialize(paths)
            print("\nServer environment initialization: PASS")
        else:
            initialize(paths)
            check(paths, args)
    except (OSError, SetupError, subprocess.SubprocessError) as exc:
        print(f"Server environment setup: FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
