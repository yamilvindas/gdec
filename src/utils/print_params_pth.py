#!/usr/bin/env python3

import pickle
import argparse

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--parameters_pth_file', required=True, help="Parameters for the experiment. It must be a pth file (not json)", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_pth_file']

    #==========================================================================#
    # Open the file
    with open(parameters_file, 'rb') as pf:
        params = pickle.load(pf)
    # Plotting the params
    for param in params:
        print("\n{}: {}".format(param, params[param]))

if __name__=='__main__':
    main()
