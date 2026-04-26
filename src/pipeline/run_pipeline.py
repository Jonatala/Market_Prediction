import subprocess

def run_step(script):
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise Exception(f"{script} failed")

def main():
    run_step("../data/fetch_data.py")
    run_step("../features/preprocess.py")
    run_step("../model_train/train.py")

if __name__ == "__main__":
    main()