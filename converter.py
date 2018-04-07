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
        if filename.split('.')[1] == 'csv':
            dframename = read_csv(filename, header=None, delimiter=',', usecols=[3, 4], engine='c')
            chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR\{}'.format(destinationfolder))
            dframename.to_csv('{}.csv'.format(nam4), header=False, index=False)
        else:
            dframename = read_csv(filename, header=None, delim_whitespace=True, engine='c')
            chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR\{}'.format(destinationfolder))
            dframename.to_csv('{}.csv'.format(nam4), header=False, index=False)
        chdir('C:\\Users\Josh\IdeaProjects\PulsedNMR\{}'.format(datafolder))


DataConvert()
