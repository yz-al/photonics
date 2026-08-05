# arch.mk for BerkeleyGW built against the conda-forge toolchain
# (gfortran + OpenMPI + ScaLAPACK + FFTW + HDF5) inside the Modal image.
# Used only when BGW_TARBALL_URL is set so the GW-BSE branch compiles. This is a
# generic starting point — a production build may need per-release tuning.

COMPFLAG  = -DGNU
PARAFLAG  = -DMPI -DOMP
MATHFLAG  = -DUSESCALAPACK -DUNPACKED -DHDF5 -DUSEFFTW3

FCPP    = /usr/bin/cpp -C -nostdinc
F90free = mpif90 -ffree-form -ffree-line-length-none -fopenmp
LINK    = mpif90 -fopenmp
FOPTS   = -O2 -funsafe-math-optimizations
FNOOPTS = $(FOPTS)
MOD_OPT = -J
INCFLAG = -I

CC_COMP = mpicc
C_COMP  = mpicc
C_LINK  = mpicc
C_OPTS  = -O2

# conda prefix provides lib/ and include/ for the numeric stack
CONDA_PREFIX ?= /opt/conda
FFTWLIB   = -L$(CONDA_PREFIX)/lib -lfftw3 -lfftw3_omp
FFTWINCLUDE = $(CONDA_PREFIX)/include
LAPACKLIB = -L$(CONDA_PREFIX)/lib -lscalapack -llapack -lblas
SCALAPACKLIB = $(LAPACKLIB)
HDF5LIB   = -L$(CONDA_PREFIX)/lib -lhdf5_fortran -lhdf5 -lz
HDF5INCLUDE = $(CONDA_PREFIX)/include

TESTSCRIPT =
