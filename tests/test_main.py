import subprocess

from example_datascience import __main__


def test_main_process():
    completed_process = subprocess.run(["python", "-m", "example_datascience"], capture_output=True, text=True)
    assert completed_process.returncode == 0


def test_main_function():
    assert __main__.main() == 0
