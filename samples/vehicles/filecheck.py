import glob

files = glob.glob('*')

n_files = len(files) - 1 
print('vehicle : ', n_files, " Out of 100 completed")
