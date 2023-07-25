from PassiveWFMeasurement import h5_storage


data_dir = '/mnt/data/data_2023-06-13/'

lasing_off_file = data_dir+'2023_06_13-21_03_40_Lasing_False_SATBD02-DSCR050.h5'
lasing_on_file = data_dir+'2023_06_13-20_58_14_Lasing_True_SATBD02-DSCR050.h5'


data = h5_storage.loadH5Recursive(lasing_off_file)



