from os import chdir, getcwd, listdir

from pandas import read_csv


def DataConvert(datafolder='DAT', destinationfolder='RDAT'):
    chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR\{}'.format(datafolder))
    for filename in listdir(getcwd()):
        name = filename.split('.')[0]
        nam2 = name.split('_')
        nam3 = '{}_'.format(nam2[0])
        for i in range(1, len(nam2) - 1):
            nam3 += '{}_'.format(nam2[i])
        nam4 = nam3 + ('R' + nam2[-1])
        if filename.split('.')[1] == 'txt':
            dframename1 = read_csv(filename, header=None, delimiter=',', engine='c', names=['t', 'M'])
            print(dframename1)
            chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR\{}'.format(destinationfolder))
            dframename1.to_hdf('{}.h5'.format(nam4), 'table', mode='w')
        if filename.split('.')[1] == 'csv':
            dframename2 = read_csv(filename, header=None, delimiter=',', usecols=[3, 4], engine='c', names=['t', 'M'])
            print(dframename2)
            chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR\{}'.format(destinationfolder))
            dframename2.to_hdf('{}.h5'.format(nam4), 'table', mode='w')
        chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR\{}'.format(datafolder))
