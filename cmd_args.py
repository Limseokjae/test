import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--keep-files", action='store_true', help="keep temporary files")
args = parser.parse_args()  