import subprocess

print("Running supervised models...")
subprocess.run(["python", "src/supervised/supervised_models.py"])

print("Running unsupervised models...")
subprocess.run(["python", "src/unsupervised/unsupervised_models.py"])
