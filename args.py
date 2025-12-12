from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser("PRO3", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")

parser.add_argument("--dataset", default="MUTAG", help="Dataset")
parser.add_argument("--method", default="Ours", help="Method")
parser.add_argument("--num", default=5000, type=int, help="Number of graphs")
parser.add_argument("--type", default="poi", help="Type of Sampler")

args = parser.parse_args()