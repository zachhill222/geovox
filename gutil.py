#!/usr/bin/env python3

# --------------------------------------------------------------
# Manage the GeometryUtility (gutil) submodule
# 
# Place this file in the root directory of a library that uses
# gutil to clone GeometryUtility into a sub-directory 
# as a submodule.
# 
# commands: install, uninstall, update, status, reinstall
# --------------------------------------------------------------



import argparse
import subprocess
import sys
import shutil
from pathlib import Path

# --------------------------------------------------------------
# Set repo and install path
# --------------------------------------------------------------
REPO = "https://github.com/zachhill222/GeometryUtility.git"
LIBNAME = "gutil"
DEFAULT_PATH = "external/" + LIBNAME
DEFAULT_BRANCH = "main"

# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------
def run (cmd: list[str], cwd: Path | None = None, check: bool =  True) -> subprocess.CompletedProcess:
	# Run a shell commend, printing it first. Exit on failure unless check=False
	print(f" $ {' '.join(cmd)}")
	result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
	
	# forward any output text or return codes to the shell
	if result.stdout.strip():
		# forward any 
		print(result.stdout.strip())
	if result.returncode != 0:
		if result.stderr.strip():
			# forward any error output to the shell stderr
			print(f"\tstderr: {result.stderr.strip()}", file=sys.stderr)
		if check:
			print(f"\n[error] command failed: {' '.join(cmd)}", file=sys.stderr)
			sys.exit(1)
	return result


def find_project_root() -> Path:
	# Walk up from cwd to the root .git directory
	cwd = Path.cwd()
	for parent in [cwd, *cwd.parents]:
		if (parent / ".git").exists():
			return parent
	print("[error] not inside a git repository.")
	sys.exit(1)


def check_git() -> None:
	# Ensure git is available
	if shutil.which("git") is None:
		print("[error] git is not installed or not in PATH.", file=sys.stderr)
		sys.exit(1)


def submodule_exists(root: Path, path: Path) -> bool:
	# Check whether the submodule is already registerd in .gitmodules
	gitmodules = root / ".gitmodules"
	if not gitmodules.exists():
		return False

	rel = path.relative_to(root)
	return str(rel) in gitmodules.read_text()


def get_tracked_branch(root: Path, rel: str) -> str | None:
	# Read the branch tracked in .gitmodules for this submodule.
	result = run(["git", "config", "--file", ".gitmodules", f"submodule.{rel}.branch"],
					cwd=root, check=False)
	return result.stdout.strip() or None

# --------------------------------------------------------------
# Primary functions
# --------------------------------------------------------------
def install(root: Path, submodule_path: Path, branch: str) -> None:
	print(f"\n[install] adding {LIBNAME} as a submodule at '{submodule_path.relative_to(root)}'.")

	# sanity/error checks
	if submodule_exists(root, submodule_path):
		print("[install] submodule already registered - run 'update' to update or 'reinstall' to reinstall.")
		return

	if submodule_path.exists() and any(submodule_path.iterdir()):
		print(f"[error] target directory '{submodule_path}' already exists and is not empty.", file=sys.stderr)
		sys.exit(1)

	rel = str(submodule_path.relative_to(root))

	# add submodule
	run(["git", "submodule", "add", "--branch", branch, REPO, rel], cwd=root)

	# initialize and fetch
	run(["git", "submodule", "update", "--init", "--recursive", rel], cwd=root)

	print(f"\n[install] done. {LIBNAME} is at '{rel}'.")


def uninstall(root: Path, submodule_path: Path) -> None:
	print(f"\n[uninstall] removing {LIBNAME} at '{submodule_path.relative_to(root)}'.")

	# sanity/error checks
	if not submodule_exists(root, submodule_path):
		print("[uninstall] submodule not found in .gitmodules - nothing to do.")
		return

	
	rel = str(submodule_path.relative_to(root))

	# remove from working tree
	run(["git", "submodule", "deinit", "--force", rel], cwd=root)
	
	# remove from index
	run(["git", "rm", "--force", rel], cwd=root)

	# remove from .git/modules
	git_modules_dir = root / ".git" / "modules" / rel
	if git_modules_dir.exists():
		print(f"\tremoving '{git_modules_dir}'")
		shutil.rmtree(git_modules_dir)

	print(f"\n[uninstall] done.")

def update(root: Path, submodule_path: Path) -> None:
	print(f"\n[update] updating {LIBNAME} at '{submodule_path.relative_to(root)}'.")

	rel = str(submodule_path.relative_to(root))
	branch = get_tracked_branch(root, rel)

	if not submodule_exists(root, submodule_path):	
		print("[error] submodule not registered - run 'install' first.", file=sys.stderr)
		sys.exit(1)

	# make sure git is initialized
	run(["git", "submodule", "update", "--init", rel], cwd=root)

	# fetch latest on the tracked branch
	run(["git", "fetch", "origin"], cwd=submodule_path)
	run(["git", "checkout", branch], cwd=submodule_path)
	run(["git", "pull", "origin", branch], cwd=submodule_path)

	# update the parent repos' recorded commit
	run(["git", "add", rel], cwd=root)

	print(f"\n[update] done.")

def status(root: Path, submodule_path: Path) -> None:
	print(f"\n[status] {LIBNAME} submodule at '{submodule_path.relative_to(root)}'.")

	if not submodule_exists(root, submodule_path):
		print("[status] submodule not registered.")
		return

	# git submodule status
	result = run(["git", "submodule", "status", str(submodule_path.relative_to(root))],
					cwd=root, check=False)
	
	# show current commit message if initialized
	if submodule_path.exists():
		run(["git", "log", "--oneline", "-1"], cwd=submodule_path, check=False)


def reinstall(root: Path, submodule_path: Path, branch: str) -> None:
	print(f"\n[reinstall] {LIBNAME} at '{submodule_path.relative_to(root)}'.")
	
	# simple uninstall followed by install
	uninstall(root, submodule_path)
	install(root, submodule_path, branch)

# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
def main() -> None:
	# get arguments
	parser = argparse.ArgumentParser(
		description = f"Manage {LIBNAME} as a git submodule.",
		formatter_class = argparse.RawDescriptionHelpFormatter,
		epilog = __doc__,
	)

	parser.add_argument(
		"command",
		choices = ["install", "uninstall", "update", "status", "reinstall"],
		help = "action to perform",
	)

	parser.add_argument(
		"--path",
		default = DEFAULT_PATH,
		help = f"submodule path relative to project root (default: {DEFAULT_PATH})",
	)

	parser.add_argument(
		"--branch",
		default = DEFAULT_BRANCH,
		help = f"branch to track (default: {DEFAULT_BRANCH})",
	)

	args = parser.parse_args()
	

	# dispatch commands
	check_git()
	root = find_project_root()
	submodule_path = (root / args.path).resolve()

	dispatch = {
		"install":		lambda: install(root, submodule_path, args.branch),
		"uninstall":	lambda: uninstall(root, submodule_path),
		"update":		lambda: update(root, submodule_path),
		"status": 		lambda: status(root, submodule_path),
		"reinstall": 	lambda: reinstall(root, submodule_path, args.branch),
	}
	dispatch[args.command]()


if __name__ == "__main__":
	main()