import argparse
from test2 import Train, Noise

def main():
	#print("executing...")
	globalargs = {'train': Train(), 'noise': Noise()}

	parser = argparse.ArgumentParser()

	global_group = parser.add_mutually_exclusive_group()
	global_group.add_argument(prog = 'train', description = 'Train a model.')
	global_group.add_argument(prog = 'noise', description = 'Run noise experiment.')


	for global_arg in globalargs:
	    globalargs[global_arg].add_args(parser)
	
	parser.parse_args()

if __name__ == '__main__':
	main()

#print("This is my file to test Python's execution methods.")
#print("The variable __name__ tells me which context this file is running in.")
#print("The value of __name__ is:", repr(__name__))