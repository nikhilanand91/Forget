import argparse

def main():

	train_parser = argparse.ArgumentParser(description='Train a model.', add_help = False)
	#noise_parser = argparse.ArgumentParser(prog = 'noise', description='Add noise to a model.', add_help = False)
	
	#group_cmds = parser.add_mutually_exclusive_group()

	#group_cmds.add_argument('--train')
	#group_cmds.add_argument('--noise')


	subparsers = train_parser.add_subparsers(help = 'sub command')


	train_subparser = subparsers.add_parser('train', help = 'train help')

	#train_parser.add_argument('--noise_level')
	train_subparser.add_argument('model')
	train_subparser.add_argument('--LR')
	train_subparser.add_argument('--dataset')
	#train_args = train_parser_child.parse_args()


	noise_subparser = subparsers.add_parser('noise', help = 'noise help')

	#noise_parser_child = argparse.ArgumentParser(parents = [noise_parser])
	noise_subparser.add_argument('model')
	noise_subparser.add_argument('--noise_level')
	noise_subparser.add_argument('--model_name')
	#noise_args = noise_parser_child.parse_args()

	args = train_parser.parse_args()
	print(args)

	#if train_args.model != None:
	#		print(f'Training model {train_args.dataset} with LR {train_args.LR}')

	#if noise_args.model != None:
	#	print(f'Doing noise experiment with noise {noise_args.noise_level} and model {noise_args.model_name}')

	

if __name__ == '__main__':
	main()