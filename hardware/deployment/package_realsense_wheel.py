"""Package an already built, static librealsense Python extension privately.

Uses the pinned upstream Python packaging files; does not install system files.
Run with the same Python interpreter used to compile the extension.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-version", help="Optional backend tag, such as v4l2, for a distinct backup wheel")
    args = parser.parse_args()
    source, build, staging, output = [p.resolve() for p in
                                     (args.source, args.build, args.staging, args.output)]
    extension_name = "pyrealsense2" + sysconfig.get_config_var("EXT_SUFFIX")
    candidates = list(build.rglob(extension_name))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {extension_name}, found {candidates}")
    # Refuse to overwrite a previous packaging run.
    staging.mkdir(parents=True, exist_ok=False)
    package = staging / "pyrealsense2"
    package.mkdir()
    upstream = source / "wrappers/python"
    for name in ("pyproject.toml", "README.rst", "hatch_build.py"):
        shutil.copy2(upstream / name, staging / name)
    shutil.copy2(upstream / "pyrealsense2/__init__.py", package / "__init__.py")
    shutil.copy2(candidates[0], package / extension_name)
    # Derive the release version from the headers used to compile this binary.
    import re
    header = (source / "include/librealsense2/rs.h").read_text()
    parts = [re.search(r"#define\s+RS2_API_" + part + r"_VERSION\s+(\d+)", header).group(1)
             for part in ("MAJOR", "MINOR", "PATCH")]
    version = ".".join(parts)
    if args.local_version:
        if not re.fullmatch(r"[a-z0-9]+(?:[.][a-z0-9]+)*",args.local_version):
            raise ValueError("Invalid wheel local-version tag")
        version += "+"+args.local_version
    (package / "_version.py").write_text(f'__version__ = "{version}"\n')
    # Upstream infers the interpreter/ABI tag; also mark the wheel as binary.
    hook = staging / "hatch_build.py"
    hook.write_text(hook.read_text().replace("build_data['infer_tag'] = True",
                                             "build_data['infer_tag'] = True\n"
                                             "        build_data['pure_python'] = False"))
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                    "--outdir", str(output), str(staging)], check=True)


if __name__ == "__main__":
    main()
