# Machine spec collection script from:
# wget https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py

# mypy: allow-untyped-defs

import argparse

# Unlike the rest of the PyTorch this file must be python2 compliant.
# This script outputs relevant system environment info
# Run it with `python collect_env.py` or `python -m torch.utils.collect_env`
import datetime
import json
import locale
import os
import re
import subprocess
import sys
from collections import namedtuple
from typing import cast as _cast

from sglang.utils import save_json

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "is_debug_build",
        "cuda_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_cuda_available",
        "cuda_runtime_version",
        "cuda_module_loading",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
        "is_xpu_available",
        "hip_compiled_version",
        "hip_runtime_version",
        "miopen_runtime_version",
        "caching_allocator_config",
        "is_xnnpack_available",
        "cpu_info",
        "system_ram",
    ],
)


def run(command):
    """Return (return-code, stdout, stderr)."""
    shell = True if type(command) is str else False
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
    )
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    if get_platform() == "win32":
        enc = "oem"
    else:
        enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def get_clang_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version (.*)"
    )


def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def get_nvidia_driver_version(run_lambda):
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(
            run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]"
        )
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")


def get_gpu_info(run_lambda):
    if get_platform() == "darwin" or (
        TORCH_AVAILABLE
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    ):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.version.hip is not None:
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "gcnArchName"):
                    gcnArch = " ({})".format(prop.gcnArchName)
                else:
                    gcnArch = "NoGCNArchNameOnOldPyTorch"
            else:
                gcnArch = ""
            return torch.cuda.get_device_name(None) + gcnArch
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)


def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def get_cudnn_version(run_lambda):
    """Return a list of libcudnn.so; it's hard to tell which one is being used."""
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        cuda_path = os.environ.get("CUDA_PATH", "%CUDA_PATH%")
        where_cmd = os.path.join(system_root, "System32", "where")
        cudnn_cmd = '{} /R "{}\\bin" cudnn*.dll'.format(where_cmd, cuda_path)
    elif get_platform() == "darwin":
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        # https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-mac-os-x/index.html#installation
        # https://docs.nvidia.com/deeplearning/cudnn/installation/latest/
        # Use CUDNN_LIBRARY when cudnn library is installed elsewhere.
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc != 1 and rc != 0):
        l = os.environ.get("CUDNN_LIBRARY")
        if l is not None and os.path.isfile(l):
            return os.path.realpath(l)
        return None
    files_set = set()
    for fn in out.split("\n"):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if os.path.isfile(fn):
            files_set.add(fn)
    if not files_set:
        return None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = sorted(files_set)
    if len(files) == 1:
        return files[0]
    result = "\n".join(files)
    return "Probably one of the following:\n{}".format(result)


def get_nvidia_smi():
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        legacy_path = os.path.join(
            program_files_root, "NVIDIA Corporation", "NVSMI", smi
        )
        new_path = os.path.join(system_root, "System32", smi)
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if os.path.exists(candidate_smi):
                smi = '"{}"'.format(candidate_smi)
                break
    return smi


def _detect_linux_pkg_manager():
    if get_platform() != "linux":
        return "N/A"
    for mgr_name in ["dpkg", "dnf", "yum", "zypper"]:
        rc, _, _ = run(f"which {mgr_name}")
        if rc == 0:
            return mgr_name
    return "N/A"


def get_linux_pkg_version(run_lambda, pkg_name):
    pkg_mgr = _detect_linux_pkg_manager()
    if pkg_mgr == "N/A":
        return "N/A"

    grep_version = {
        "dpkg": {
            "field_index": 2,
            "command": "dpkg -l | grep {}",
        },
        "dnf": {
            "field_index": 1,
            "command": "dnf list | grep {}",
        },
        "yum": {
            "field_index": 1,
            "command": "yum list | grep {}",
        },
        "zypper": {
            "field_index": 2,
            "command": "zypper info {} | grep Version",
        },
    }

    field_index: int = int(_cast(int, grep_version[pkg_mgr]["field_index"]))
    cmd: str = str(grep_version[pkg_mgr]["command"])
    cmd = cmd.format(pkg_name)
    ret = run_and_read_all(run_lambda, cmd)
    if ret is None or ret == "":
        return "N/A"
    lst = re.sub(" +", " ", ret).split(" ")
    if len(lst) <= field_index:
        return "N/A"
    return lst[field_index]


def get_intel_gpu_driver_version(run_lambda):
    lst = []
    platform = get_platform()
    if platform == "linux":
        pkgs = {  # type: ignore[var-annotated]
            "dpkg": {
                "intel-opencl-icd",
                "libze1",
                "level-zero",
            },
            "dnf": {
                "intel-opencl",
                "level-zero",
            },
            "yum": {
                "intel-opencl",
                "level-zero",
            },
            "zypper": {
                "intel-opencl",
                "level-zero",
            },
        }.get(_detect_linux_pkg_manager(), {})
        for pkg in pkgs:
            ver = get_linux_pkg_version(run_lambda, pkg)
            if ver != "N/A":
                lst.append(f"* {pkg}:\t{ver}")
    if platform in ["win32", "cygwin"]:
        txt = run_and_read_all(
            run_lambda,
            'powershell.exe "gwmi -Class Win32_PnpSignedDriver | where{$_.DeviceClass -eq \\"DISPLAY\\"\
            -and $_.Manufacturer -match \\"Intel\\"} | Select-Object -Property DeviceName,DriverVersion,DriverDate\
            | ConvertTo-Json"',
        )
        try:
            obj = json.loads(txt)
            if type(obj) is list:
                for o in obj:
                    lst.append(
                        f'* {o["DeviceName"]}: {o["DriverVersion"]} ({o["DriverDate"]})'
                    )
            else:
                lst.append(f'* {obj["DriverVersion"]} ({obj["DriverDate"]})')
        except ValueError as e:
            lst.append(txt)
            lst.append(str(e))
    return "\n".join(lst)


def get_intel_gpu_onboard(run_lambda):
    lst: list[str] = []
    platform = get_platform()
    if platform == "linux":
        txt = run_and_read_all(run_lambda, "xpu-smi discovery -j")
        if txt:
            try:
                obj = json.loads(txt)
                device_list = obj.get("device_list", [])
                if isinstance(device_list, list) and device_list:
                    lst.extend(f'* {device["device_name"]}' for device in device_list)
                else:
                    lst.append("N/A")
            except (ValueError, TypeError) as e:
                lst.append(txt)
                lst.append(str(e))
        else:
            lst.append("N/A")
    if platform in ["win32", "cygwin"]:
        txt = run_and_read_all(
            run_lambda,
            'powershell.exe "gwmi -Class Win32_PnpSignedDriver | where{$_.DeviceClass -eq \\"DISPLAY\\"\
            -and $_.Manufacturer -match \\"Intel\\"} | Select-Object -Property DeviceName | ConvertTo-Json"',
        )
        if txt:
            try:
                obj = json.loads(txt)
                if isinstance(obj, list) and obj:
                    lst.extend(f'* {device["DeviceName"]}' for device in obj)
                else:
                    lst.append(f'* {obj.get("DeviceName", "N/A")}')
            except ValueError as e:
                lst.append(txt)
                lst.append(str(e))
        else:
            lst.append("N/A")
    return "\n".join(lst)


def get_intel_gpu_detected(run_lambda):
    if not TORCH_AVAILABLE or not hasattr(torch, "xpu"):
        return "N/A"

    device_count = torch.xpu.device_count()
    if device_count == 0:
        return "N/A"

    devices = [
        f"* [{i}] {torch.xpu.get_device_properties(i)}" for i in range(device_count)
    ]
    return "\n".join(devices)


def get_cpu_info(run_lambda):
    rc, out, err = 0, "", ""
    if get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
    elif get_platform() == "win32":
        rc, out, err = run_lambda(
            'powershell.exe "gwmi -Class Win32_Processor | Select-Object -Property Name,Manufacturer,Family,\
            Architecture,ProcessorType,DeviceID,CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision\
            | ConvertTo-Json"'
        )
        if rc == 0:
            lst = []
            try:
                obj = json.loads(out)
                if type(obj) is list:
                    for o in obj:
                        lst.append("----------------------")
                        lst.extend([f"{k}: {v}" for (k, v) in o.items()])
                else:
                    lst.extend([f"{k}: {v}" for (k, v) in obj.items()])
            except ValueError as e:
                lst.append(out)
                lst.append(str(e))
            out = "\n".join(lst)
    elif get_platform() == "darwin":
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    cpu_info = "None"
    if rc == 0:
        cpu_info = out
    else:
        cpu_info = err
    return cpu_info


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("cygwin"):
        return "cygwin"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    else:
        return sys.platform


def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def get_windows_version(run_lambda):
    ret = run_and_read_all(
        run_lambda,
        'powershell.exe "gwmi -Class Win32_OperatingSystem | Select-Object -Property Caption,\
        OSArchitecture,Version | ConvertTo-Json"',
    )
    try:
        obj = json.loads(ret)
        ret = f'{obj["Caption"]} ({obj["Version"]} {obj["OSArchitecture"]})'
    except ValueError as e:
        ret += f"\n{str(e)}"
    return ret


def get_lsb_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "lsb_release -a", r"Description:\t(.*)"
    )


def check_release_file(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"'
    )


def get_os(run_lambda):
    from platform import machine

    platform = get_platform()

    if platform in ["win32", "cygwin"]:
        return get_windows_version(run_lambda)

    if platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return "macOS {} ({})".format(version, machine())

    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        return "{} ({})".format(platform, machine())

    # Unknown platform
    return platform


def get_python_platform():
    import platform

    return platform.platform()


def get_libc_version():
    import platform

    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def get_cachingallocator_config():
    ca_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if not ca_config:
        ca_config = os.environ.get("PYTORCH_HIP_ALLOC_CONF", "")
    return ca_config


def get_cuda_module_loading_config():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.init()
        config = os.environ.get("CUDA_MODULE_LOADING", "")
        return config
    else:
        return "N/A"


def is_xnnpack_available():
    if TORCH_AVAILABLE:
        import torch.backends.xnnpack

        return str(torch.backends.xnnpack.enabled)  # type: ignore[attr-defined]
    else:
        return "N/A"


def get_system_ram(run_lambda):
    platform = get_platform()
    if platform == "linux":
        rc, out, err = run_lambda("grep 'MemTotal' /proc/meminfo")
        if rc == 0:
            return out.strip()
    elif platform == "darwin":
        rc, out, err = run_lambda("sysctl -n hw.memsize")
        if rc == 0:
            try:
                ram_bytes = int(out)
                ram_gb = ram_bytes / (1024**3)
                return f"{ram_gb:.2f} GB"
            except ValueError:
                return out.strip()
    elif platform == "win32":
        rc, out, err = run_lambda("wmic ComputerSystem get TotalPhysicalMemory")
        if rc == 0:
            try:
                ram_bytes = int(out.split("\n")[1].strip())
                ram_gb = ram_bytes / (1024**3)
                return f"{ram_gb:.2f} GB"
            except (ValueError, IndexError):
                return out.strip()
    return "N/A"


def get_env_info():
    run_lambda = run

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
        xpu_available_str = str(torch.xpu.is_available())
        if torch.xpu.is_available():
            xpu_available_str = (
                f"{xpu_available_str}\n"
                + f"XPU used to build PyTorch: {torch.version.xpu}\n"
                + f"Intel GPU driver version:\n{get_intel_gpu_driver_version(run_lambda)}\n"
                + f"Intel GPU models onboard:\n{get_intel_gpu_onboard(run_lambda)}\n"
                + f"Intel GPU models detected:\n{get_intel_gpu_detected(run_lambda)}"
            )
        if (
            not hasattr(torch.version, "hip") or torch.version.hip is None
        ):  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"
        else:  # HIP version

            def get_version_or_na(cfg, prefix):
                _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]
                return _lst[0] if _lst else "N/A"

            cfg = torch._C._show_config().split("\n")
            hip_runtime_version = get_version_or_na(cfg, "HIP Runtime")
            miopen_runtime_version = get_version_or_na(cfg, "MIOpen")
            cuda_version_str = "N/A"
            hip_compiled_version = torch.version.hip
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = (
            xpu_available_str
        ) = "N/A"
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"

    sys_version = sys.version.replace("\n", " ")

    return SystemEnv(
        torch_version=version_str,
        is_debug_build=debug_mode_str,
        python_version="{} ({}-bit runtime)".format(
            sys_version, sys.maxsize.bit_length() + 1
        ),
        python_platform=get_python_platform(),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        cuda_module_loading=get_cuda_module_loading_config(),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        is_xpu_available=xpu_available_str,
        hip_compiled_version=hip_compiled_version,
        hip_runtime_version=hip_runtime_version,
        miopen_runtime_version=miopen_runtime_version,
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        caching_allocator_config=get_cachingallocator_config(),
        is_xnnpack_available=is_xnnpack_available(),
        cpu_info=get_cpu_info(run_lambda),
        system_ram=get_system_ram(run_lambda),
    )


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Collect environment information.")
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Dir to save the collected environment information as a JSON file.",
    )
    return vars(parser.parse_args())


def main():
    config = parse_args()
    print("Collecting environment information...")
    env_info = get_env_info()

    # Create a dictionary with relevant sections
    output_dict = {
        "pytorch_info": {
            "torch_version": env_info.torch_version,
            "is_debug_build": env_info.is_debug_build,
            "cuda_compiled_version": env_info.cuda_compiled_version,
            "hip_compiled_version": env_info.hip_compiled_version,
            "is_xnnpack_available": env_info.is_xnnpack_available,
        },
        "system_info": {
            "os": env_info.os,
            "libc_version": env_info.libc_version,
            "gcc_version": env_info.gcc_version,
            "clang_version": env_info.clang_version,
            "cmake_version": env_info.cmake_version,
        },
        "python_info": {
            "python_version": env_info.python_version,
            "python_platform": env_info.python_platform,
        },
        "cuda_info": {
            "is_cuda_available": env_info.is_cuda_available,
            "cuda_runtime_version": env_info.cuda_runtime_version,
            "cuda_module_loading": env_info.cuda_module_loading,
            "nvidia_driver_version": env_info.nvidia_driver_version,
            "nvidia_gpu_models": env_info.nvidia_gpu_models,
            "cudnn_version": env_info.cudnn_version,
        },
        "xpu_info": {
            "is_xpu_available": env_info.is_xpu_available,
        },
        "rocm_info": {
            "hip_runtime_version": env_info.hip_runtime_version,
            "miopen_runtime_version": env_info.miopen_runtime_version,
        },
        "cpu_info": {
            "cpu_info": env_info.cpu_info,
        },
        "memory_info": {
            "system_ram": env_info.system_ram,
            "caching_allocator_config": env_info.caching_allocator_config,
        },
    }

    # Print the dictionary as a JSON object
    print(json.dumps(output_dict, indent=4))
    os.makedirs(config["out_dir"], exist_ok=True)
    save_json(f"{config['out_dir']}/machine_specs.json", output_dict)


if __name__ == "__main__":
    main()
