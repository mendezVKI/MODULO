from tqdm import tqdm
import numpy as np
import os
import glob
from multiprocessing import Pool
from itertools import product
from functools import wraps
from time import time, sleep

def write_matrix_to_textfile(a_matrix, file_to_write):
 def compile_row_string(a_row):
  return str(a_row).strip(']').strip('[').replace(' ', '')

 with open(file_to_write, 'w') as f:
  for row in a_matrix:
   f.write(compile_row_string(row) + '\n')

 return True

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format(end-start))
        return result
    return wrapper


def load_from_columns_single_file(file, folder, columns_in_order, skip_lines):
  DATA = np.genfromtxt(folder + os.sep + file, skip_header=skip_lines)  # Here we have the four colums
  columns_in_order = list(columns_in_order)
  D3 = DATA[:, columns_in_order] # grab the selected columns
  D3 = D3.flatten('F') # flatten the result into one vector
  return D3


def load_from_columns_parallel(pattern_file,folder, columns_in_order, skip_lines, timesteps = None):
  files_to_load = [os.path.basename(f) for f in sorted(glob.glob(folder + os.sep + '*'+pattern_file+'*'))]
  if timesteps is not None:
    files_to_load = files_to_load[:timesteps] # select only the first n_t files

  cpu_count = os.cpu_count() # count cpus to create the pool

  print('Loading files in parallel using: ' + str(cpu_count) +' processes')

  job_args = [(filename, folder, columns_in_order, skip_lines) for filename in files_to_load]
  with Pool(processes=cpu_count) as pool:
     result = pool.map(auxiliary_function_parallel, job_args)


  return result

def auxiliary_function_parallel(args):
  # this function is just an auxiliary function that is used pass the arguments to the actual function that performs the computations.
  return load_from_columns_single_file(*args)


def load_from_columns(D_init, pattern_file,folder, skip_lines, columns_in_order):

  files_to_load = [os.path.basename(f) for f in sorted(glob.glob(folder + os.sep + '*'+pattern_file+'*'))]

  # print(files_to_load)
  for k, file in enumerate(tqdm(files_to_load)):
    # Read data from a file
    DATA = np.genfromtxt(folder + os.sep + file, skip_header=skip_lines) # Here we have the four colums
    D1 = DATA[:,columns_in_order]
    D_init[:, k] = np.concatenate(D1, axis=0)  # Reshape and assign

  return D_init

