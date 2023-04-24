import subprocess
import os
import sys
import shlex
import importlib.util

commandline_args = os.environ.get("COMMANDLINE_ARGS", "")
sys.argv += shlex.split(commandline_args)

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
stored_commit_hash = None
skip_install = False


def run(command, desc=None, errdesc=None, custom_env=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ if custom_env is None else custom_env,
    )

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def check_run(command):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    return result.returncode == 0


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f" --index-url {index_url}" if index_url != "" else ""
    return run(
        f'"{python}" -m pip {args} --prefer-binary{index_url_line}',
        desc=f"Installing {desc}",
        errdesc=f"Couldn't install {desc}",
    )


def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


def fix_faiss():
    spec = importlib.util.find_spec("faiss")
    if (
        spec.submodule_search_locations is None
        or len(spec.submodule_search_locations) == 0
    ):
        return
    dir = spec.submodule_search_locations[0]
    if os.path.exists(os.path.join(dir, "swigfaiss_avx2.py")):
        return
    try:
        os.symlink(
            os.path.join(dir, "swigfaiss.py"), os.path.join(dir, "swigfaiss_avx2.py")
        )
    except:
        pass


def prepare_environment():
    commit = commit_hash()

    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")

    torch_command = os.environ.get(
        "TORCH_COMMAND",
        "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118",
    )

    sys.argv, skip_install = extract_arg(sys.argv, "--skip-install")
    if skip_install:
        return

    sys.argv, reinstall_torch = extract_arg(sys.argv, "--reinstall-torch")
    ngrok = "--ngrok" in sys.argv

    if reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(
            f'"{python}" -m {torch_command}',
            "Installing torch and torchvision",
            "Couldn't install torch",
        )

    if not is_installed("pyngrok") and ngrok:
        run_pip("install pyngrok", "ngrok")

    run(
        f'"{python}" -m pip install -r requirements.txt',
        desc=f"Installing requirements",
        errdesc=f"Couldn't install requirements",
    )

    fix_faiss()


def start():
    os.environ["PATH"] = (
        os.path.join(os.path.dirname(__file__), "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    subprocess.run(
        [python, "webui.py", *sys.argv[1:]],
    )


if __name__ == "__main__":
    prepare_environment()
    start()
