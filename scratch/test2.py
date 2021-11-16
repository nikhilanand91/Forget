import argparse

class Train:
	def __init__(self):
		pass

	def add_args(self, parser: argparse.ArgumentParser):
		allowed_params = {'--lr': float, '--save': bool}

		#print('keys:')
		#print(allowed_params.keys())

		for param in allowed_params.keys():
			parser.add_argument(param,
							    type = allowed_params[param],
							    help = 'help text')

class Noise:
	def __init__(self):
		pass

	def add_args(self, parser: argparse.ArgumentParser):
		allowed_params = {'--noise_level': float, '--save': bool}

		#print('keys:')
		#print(allowed_params.keys())

		for param in allowed_params.keys():
			parser.add_argument(param,
							    type = allowed_params[param],
							    help = 'help text')

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("train")
args = parser.parse_args()
print(args.echo)

import argparse


parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--foo', action='store_true', help='foo help')
subparsers = parser.add_subparsers(help='sub-command help')

# create the parser for the "a" command
parser_a = subparsers.add_parser('a', help='a help')
parser_a.add_argument('bar', type=int, help='bar help')

# create the parser for the "b" command
parser_b = subparsers.add_parser('b', help='b help')
parser_b.add_argument('--baz', choices='XYZ', help='baz help')

# parse some argument lists
parser.parse_args(['a', '12'])

parser.parse_args(['--foo', 'b', '--baz', 'Z'])
"""