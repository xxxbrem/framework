# Project HashClash - MD5 & SHA-1 cryptanalytic toolbox

We use the code from https://github.com/cr-marcstevens/hashclash to generate MD5 collisions. 

## Requirements

- C++11 compiler (e.g. g++)
- make
- autoconf & automake & libtool

  `sudo apt-get install autoconf automake libtool`
  
- zlib & bzip2 libraries

  `sudo apt-get install zlib1g-dev libbz2-dev`
  
- Optional: CUDA
  
## Building (automatic)

- Run build.sh script

   `./build.sh`

## Building (manual)

<details>
<summary>See details</summary>
<p>
  
- local boost C++ libraries (preferable version 1.57.0)

  `./install_boost.sh` 

  Override default boost version 1.57.0 and/or installation directory as follows:
  
  `BOOST_VERSION=1.65.1 BOOST_INSTALL_PREFIX=$HOME/boost/boost-1.65.1 ./install_boost.sh`
  
- Build configure script

  `autoreconf --install`
  
- Run configure (with boost installed in `$(pwd)/boost-VERSION` by `install_boost.sh`)

  `./configure --with-boost=$(pwd)/boost-1.57.0 [--without-cuda|--with-cuda=/usr/local/cuda-X.X]`

- Build programs

  `make [-j 4]`

</p>
</details>

## Create your own chosen-prefix collisions

Put the clean file and poisoned file in `./cpc_workdir/cpc_workdir` (eg: model_clean.bin, model_poisoned.bin)

Run script `./cpc_workdir/run.sh` to get 50 collisions.

## Create your own identical-prefix collision

Put the prefix in `./ipc_workdir/ipc_workdir` (eg: prefix.txt)

Run script `./ipc_workdir/run.sh` to get 50 collisions. 