#!/usr/bin/env python

""" run a single structure tensor analysis test on a user inputted phantom.
    Save a single row of a pandas dataframe to out.
"""

import argparse
import os
import numpy as np
import pandas as pd
import pickle
# from fibermetric import sta_validate
import sta_validate
import tqdm

derivative_sigmas = np.linspace(start=0.15, stop=2.5, num=10)
tensor_sigmas = np.linspace(start=0.0, stop=5.0, num=10)

def main(path, out):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)]
    else:
        files = [path]

    for file in tqdm.tqdm(files):
        npz = np.load(file)
        phantom = npz['image']
        AI = npz['AI']
        period = npz['period']
        angle = npz['angle']

        # save dataframe using pickle
        if phantom.ndim == 2:
            if angle.ndim == 0:
                name = f'error_AI-{AI:.2f}_period-{period:02d}_theta-{angle:.2f}.p'
            else:
                name = f'error_AI-{AI:.2f}_period-{period:02d}_theta-{angle[0]:.2f}-{angle[1]:.2f}.p'

        if phantom.ndim == 3:
            if angle.ndim == 1:
                    name = f'error_AI-{AI:.2f}_period-{period:02d}_theta-{angle[0]:.2f}_phi-{angle[1]:.2f}.p'
            else:
                name = f'error_AI-{AI:.2f}_period-{period:02d}_theta-{angle[0,0]:.2f}-{angle[1,0]:.2f}_phi-{angle[0,1]:.2f}-{angle[1,1]:.2f}.p'
        
        if os.path.exists(os.path.join(out,name)):
            continue

        error_df = pd.DataFrame({'derivative_sigma':[], 'tensor_sigma':[], 'AI':[], 'period':[],
                                'width':[], 'angles':[], 'error':[]})

        for sigma0 in derivative_sigmas:
            for sigma1 in tensor_sigmas:
                crop_all = round(max(sigma0,sigma1)*8/3) # two-thirds the radius of the largest kernel
                crop_end = round(float(AI)) - 1
                error = sta_validate.sta_test(phantom, sigma0, sigma1, true_thetas=angle, crop=crop_all, crop_end=crop_end)
                new_row = {'derivative_sigma': sigma0, 'tensor_sigma': sigma1,
                            'AI': AI, 'period': period, 'width': 1,
                            'angles': [angle], 'error': error
                        }
                error_df = pd.concat((error_df, pd.DataFrame(new_row)), ignore_index=True)

        
        # print(f'saving {name} to {out}')

        with open(os.path.join(out,name), 'wb') as f:
            pickle.dump(error_df, f)
        
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the npz file or directory of npz files")
    parser.add_argument("-o", "--out", help="output directory")
    args = parser.parse_args()

    path = args.path

    if args.out:
        out = args.out
        if not os.path.exists(out):
            os.makedirs(out)
    else:
        out = os.getcwd()

    main(path, out)