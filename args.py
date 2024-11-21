from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser("HP", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")

parser.add_argument("--dataset", default="MUTAG", help="Dataset")
parser.add_argument("--type", default="poi", help="Type of Sampler")

args = parser.parse_args()