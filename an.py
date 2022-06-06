import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', type=int, default=8)
parser.add_argument('--epoch', '-e', type=int, default=15)
parser.add_argument('--time', '-t', type=int, default=20,
                        help='Total simulation time steps.')
parser.add_argument('--rec', '-r', action='store_true' ,default=False)  # -r付けるとTrue                  
parser.add_argument('--forget', '-f', action='store_true' ,default=False) 
parser.add_argument('--dual', '-d', action='store_true' ,default=False)
args = parser.parse_args()

print(args.t)