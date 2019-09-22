import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model")

args = parser.parse_args()


print(args.model)
