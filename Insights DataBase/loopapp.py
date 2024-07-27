import subprocess
import time

def run_script():
    try:
        # Replace 'main.py' with the path to your actual script
        subprocess.run(['python', 'main.py'], check=True)
        print("Script completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Script encountered an error: {e}")

# Infinite loop to continuously run the script
while True:
    run_script()
    print("Retrying script in 5 seconds...")
    time.sleep(5)  # Wait for 5 seconds before retrying
