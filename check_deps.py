"""
Dependency checker for p4_3d.py.
Verifies required packages are installed and meet version constraints;
installs or upgrades any that are missing or out-of-range via pip.
"""

import importlib.util
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError

# (pip_name, import_name, min_ver, max_ver_exclusive, pip_install_spec)
#   min_ver / max_ver_exclusive : None means no bound
#   pip_install_spec            : string passed to pip (can include version pin)
REQUIRED = [
    ("numpy",      "numpy",      "1.20.0", None,     "numpy"),
    # cvxpy <1.3 imports scipy.misc.logsumexp, removed in scipy 1.0
    ("cvxpy",      "cvxpy",      "1.3.0",  None,     "cvxpy"),
    # ("ecos",       "ecos",       None,     None,     "ecos"),      # p4_3d.py未使用ecos求解器
    # ("clarabel",   "clarabel",   None,     None,     "clarabel"),  # clarabel随cvxpy自动安装，无需单独检查
    # ("Mosek",      "mosek",      None,     None,     "mosek"),     # 暂时用不到mosek了，兼容性很麻烦（需要手动配置license）
    ("matplotlib", "matplotlib", "3.3.0",  None,     "matplotlib"),
]

def _ver_tuple(ver_str):
    parts = []
    for seg in ver_str.split(".")[:3]:
        try:
            parts.append(int(seg.split("-")[0].split("a")[0].split("b")[0].split("rc")[0]))
        except ValueError:
            parts.append(0)
    return tuple(parts)

def check_package(pip_name, import_name, min_ver, max_ver):
    """Returns (installed, needs_action, installed_ver, reason)."""
    if importlib.util.find_spec(import_name) is None:
        return False, True, None, "missing"
    try:
        installed_ver = version(pip_name)
    except PackageNotFoundError:
        return False, True, None, "missing"

    iv = _ver_tuple(installed_ver)
    if min_ver and iv < _ver_tuple(min_ver):
        return True, True, installed_ver, f"too old (>= {min_ver} required)"
    if max_ver and iv >= _ver_tuple(max_ver):
        return True, True, installed_ver, f"too new (< {max_ver} required)"
    return True, False, installed_ver, "ok"

def pip_install(pip_spec, upgrade=False):
    cmd = [sys.executable, "-m", "pip", "install", pip_spec]
    if upgrade:
        cmd.append("--upgrade")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip()

def main():
    to_fix = []   # (pip_name, pip_spec, upgrade, reason)
    print("Checking dependencies...\n")

    for pip_name, import_name, min_ver, max_ver, pip_spec in REQUIRED:
        installed, needs_action, installed_ver, reason = check_package(
            pip_name, import_name, min_ver, max_ver)

        if needs_action:
            label = "MISSING" if not installed else "BAD VER"
            ver_info = f"  (installed: {installed_ver}, {reason})" if installed_ver else ""
            print(f"  {label}   {pip_name}{ver_info}")
            to_fix.append((pip_name, pip_spec, installed, reason))
        else:
            print(f"  OK        {pip_name}  ({installed_ver})")

    print()

    if not to_fix:
        print("All dependencies are satisfied. Nothing to install.")
        return

    print(f"Fixing {len(to_fix)} package(s)...\n")
    all_ok = True
    for pip_name, pip_spec, already_installed, reason in to_fix:
        action = "Reinstalling" if already_installed else "Installing"
        print(f"  {action} {pip_spec}  [{reason}]")
        ok, err = pip_install(pip_spec, upgrade=already_installed)
        if ok:
            print(f"    Done.")
        else:
            print(f"    FAILED:\n    {err}")
            all_ok = False

    print()
    if all_ok:
        print("All done. You can now run p4_3d.py.")
    else:
        print("Some packages failed. Check the errors above.")

if __name__ == "__main__":
    main()
