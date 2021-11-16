import argparse
import sys
from training.train_runner import TrainRunner

def main():
	"""
	Main interface. Here we parse the global arguments like 'train', 'noise', etc.
	We also add the subarguments associated with each global argument. These
	are then handled by the appropriate runner, which kicks off the job.
	"""
	global_args = {'train': TrainRunner}
	parser = argparse.ArgumentParser()
	
	subparsers = parser.add_subparsers()
	subparser_dict = {}
	for arg in global_args:
		subparser_dict[arg] = subparsers.add_parser(arg)
		subparser_dict[arg].set_defaults(func = global_args[arg].create_from_args)
		global_args[arg].add_args(subparser_dict[arg])


	args = parser.parse_args()
	if not vars(args):
		parser.print_help()
		sys.exit(0)

	runner = args.func(args)
	runner.run()

if __name__ == '__main__':
	main()