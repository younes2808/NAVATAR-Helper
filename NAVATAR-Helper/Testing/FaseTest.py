import subprocess
import time
import sys

def run_fase1():
    print("Starting fase1.py...")
    try:
        # Run fase1.py as a subprocess
        subprocess.run(['python3', 'fase1.py'], check=True)
        print("Fase1.py completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running fase1.py: {e}")
        sys.exit(1)  # Exit with error code
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)  # Exit with error code

def run_fase2():
    print("Waiting for 20 seconds before running fase2.py...")
    time.sleep(20)  # Wait for 20 seconds
    print("Starting fase2.py...")
    try:
        # Run fase2.py as a subprocess
        subprocess.run(['python3', 'fase2.py'], check=True)
        print("Fase2.py completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running fase2.py: {e}")
        sys.exit(1)  # Exit with error code
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)  # Exit with error code
        
def run_fase3():
    print("Waiting for 20 seconds before running fase3.py...")
    time.sleep(20)  # Wait for 60 seconds
    print("Starting fase3.py...")
    try:
        # Run fase2.py as a subprocess
        subprocess.run(['python3', 'fase3.py'], check=True)
        print("Fase3.py completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running fase3.py: {e}")
        sys.exit(1)  # Exit with error code
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)  # Exit with error code

if __name__ == "__main__":
    run_fase1()  # Run fase1
    run_fase2()  # After fase1 completes, automatically run fase2
    run_fase3()  # After fase2 completes, automatically run fase3