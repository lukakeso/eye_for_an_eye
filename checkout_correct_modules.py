import os
import subprocess

repo_root = os.getcwd()
# Define the repository path and commit hash
sam_hq_dir = os.path.join(repo_root, "sam-hq")
chamfer_dir = os.path.join(repo_root, "ChamferDistancePytorch")

sam_hq_commit_hash = "ac19724c47b13689e5d9596277a6522b371001c8"
chamfer_distance_commit_hash = "364c03c4ec5febc1e21068ffac362eca4a8f61d9"

# Function to run a shell command
def run_command(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, shell=True, text=True, capture_output=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("Error:", result.stderr)

# Change to the repo directory and checkout the commit
run_command("git submodule update --init --recursive", cwd=repo_root)

run_command(f"git checkout {sam_hq_commit_hash}", cwd=sam_hq_dir)

run_command(f"git checkout {chamfer_distance_commit_hash}", cwd=chamfer_dir)

run_command("mv sam-hq sam_hq", cwd=repo_root)
