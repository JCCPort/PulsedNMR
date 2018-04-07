from tkinter import filedialog

from pandas import read_csv


def DataConvert():
    datafolder = filedialog.askopenfilenames(initialdir="C:\\Users\Josh\IdeaProjects\PulsedNMR",
                                             title="Select data to convert")
    destinationfolder = filedialog.askdirectory(initialdir="C:\\Users\Josh\IdeaProjects\PulsedNMR",
                                                title="Select folder to save converted data to")
    for filename in datafolder:
        print(filename)
        name_ext = filename.split('/')[-1]
        name = name_ext.split('.')[0]
        nam2 = name.split('_')
        nam3 = '{}_'.format(nam2[0])
        for i in range(1, len(nam2) - 1):
            nam3 += '{}_'.format(nam2[i])
        nam4 = nam3 + ('R' + nam2[-1])
        if filename.split('.')[1] == 'csv':
            dframename = read_csv(filename, header=None, delimiter=',', usecols=[3, 4], engine='c')
        else:
            dframename = read_csv(filename, header=None, delim_whitespace=True, engine='c')
        print(destinationfolder)
        dframename.to_csv('{}/{}.csv'.format(destinationfolder, nam4), header=False, index=False)
