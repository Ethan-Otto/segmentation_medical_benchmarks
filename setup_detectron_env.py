import sys
import os
import subprocess
import re

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash")
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(error.decode())
        sys.exit(1)
    return output.decode()

def clean_requirement(req):
    req = re.split(r';', req)[0].strip()
    req = re.split(r'[<>=!~]', req)[0].strip()
    return req

def main():
    env_name = "detectron2_env"

    conda_path = run_command("which conda").strip()
    conda_dir = os.path.dirname(os.path.dirname(conda_path))

    init_cmd = f". {conda_dir}/etc/profile.d/conda.sh"

    print(f"Creating conda environment: {env_name}")
    run_command(f"{init_cmd} && conda create -n {env_name} python=3.8 -y")

    def run_in_env(cmd):
        return run_command(f"{init_cmd} && conda activate {env_name} && {cmd}")

    print("Installing build tools and compilers...")
    run_in_env("conda install -y conda-build")
    run_in_env("conda install -y gcc_linux-64 gxx_linux-64")

    print("Installing pyyaml...")
    run_in_env("python -m pip install pyyaml==5.1")


    if not os.path.exists('detectron2'):
            
        print("Cloning Detectron2 repository...")
        run_command("git clone https://github.com/facebookresearch/detectron2")

    print("Installing required packages...")
    with open("detectron2/setup.py", "r") as f:
        setup_content = f.read()
    install_requires = eval(setup_content.split("install_requires=")[1].split("]")[0] + "]")
    cleaned_requires = [clean_requirement(req) for req in install_requires]
    install_requires_str = ' '.join(cleaned_requires)
    run_in_env(f"python -m pip install {install_requires_str}")

    print("Installing PyTorch and torchvision...")
    run_in_env("conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y")

    print("Installing Detectron2...")
    run_in_env("python -m pip install -e detectron2")

    print("Installing axillary packages")
    run_in_env("pip install opencv-python")

    print("Setup complete!")
    print(f"To activate this environment, use: conda activate {env_name}")

if __name__ == "__main__":
    main()