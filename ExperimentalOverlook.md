Experiments to carry out:

- Measurement of spin-lattice relaxation time T<sub>1</sub> on various
 samples. This is to be done crudely by measuring change in FID amplitude
 as a function of P, the time delay between repititions of complete 
 pulse sequences. This seems to be done using a two-pulse sequence.

- Measurement of spin-spin relaxation time T<sub>2</sub> and T<sub>1</sub>
 of at least a single sample using a two-pulse sequence. The amplitude of
 echo is plotted as a function of changing the delay time 2Tau. Not yet
 sure how T<sub>2</sub> is calculated from this plot. Most accurate way
 to measure the amplitude is by fitting a curve, it seems that the echo
 is an exponential rise and fall, check this. The function echo_fits returns
 T<sub>1</sub> from the envelope function.

- Single pulse experiments. The FID of protons is dependent on their
environment. Consider a molecule with multiple atoms, the protons in
the molecule will be in different environments, consequently their 
FIDs will be different. In a sample containing multiple, similarly aligned,
molecules, these environments will be repeated resulting in the overall
FID of the sample being a superposition of the individual environments'
(sites) FIDs. Using a Fourier transform the overall signal can be broken
down into its various frequency compenents, each representing a unique
site. The code to do this is contained within main.py and the file
 'AAGlycerol_DAT_DAT_DAT.csv' in RDAT is good quality raw data from ChemSpider that
 shows well the effect of the Fourier transform. The other data currently
 contained in this folder does not seem to be the type designed for Fourier
 transforms and relates to the other two experiments above. 3 other more complicated
raw data files are also included.
 
- 1D Imaging.

