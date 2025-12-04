# robobee3d

Robobee 3D sim (and maybe files for planar ones as well)

## Installing

Common: https://github.com/avikde/controlutils

Pybullet 3D sim:
- `pip install pybullet`: need version 2.8.2 or above (see https://github.com/bulletphysics/bullet3/issues/2152 and https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=9&t=12998). It may need some VS stuff on windows and takes a while to build, but works fine.
- Don't forget to run cmake. Do that in the uprightmpc2 directory 

## Running

Pybullet 3D sim
- Generate the urdf by running `python xacro.py sdab.xacro > sdab.xacro.urdf` in the `urdf/` subdirectory (or run the task in VS code)
- For the pybullet example, run `python pybullet_hello.py`
  - `cmake -S . -B build \
  -DPython3_EXECUTABLE="/Users/franklinho/Documents/Work, Research, Finance/robobee3d/bin/python"`
  - Actually its `cd "/Users/franklinho/Documents/Work, Research, Finance/Harvard/robobee3d/template/uprightmpc2"
rm -rf build
cmake -S . -B build \
  -DPython3_EXECUTABLE="/Users/franklinho/Documents/Work, Research, Finance/robobee3d/bin/python"
cmake --build build --config Release`
- clone the github repo for `pibind11` into a "controls" folder that is same level as robobee3d 
    - do the same for eigen 
  - actually its really 

## Franklin's Setup Comments 
- First, install pybind11 into `/../../../controls/thirdparty/`. This is specified in the `CMakeLists.txt` file.
- Second, install Eigen into `/../../../SDK/thirdparty/`. This is also specified in the `CMake` file.
- Then run the below code. For some reason it likes CPP to be version 14, so that's why that line is there in the middle. 
```
cd "/Users/franklinho/Documents/Work, Research, Finance/Harvard/robobee3d/template/uprightmpc2"
rm -rf build

cmake -S . -B build \
  -DPython3_EXECUTABLE="/Users/franklinho/Documents/Work, Research, Finance/robobee3d/bin/python" \
  -DCMAKE_CXX_STANDARD=14

cmake --build build --config Release`
```
- Once the packages are all set, there are some more changes made since several packages have been updated since 2020,
when this code was last compiled. You can find them recorded in `SetupJournal.md`


## Contributing

- Create a branch and push to it
- Create a pull request to merge
- master should always have working code for everyone to use

# Julia

- Download from official or use choco
- install Ipopt.jl -- have to do `Pkg.build("Ipopt")` after following the instructions.
- try the hs071.jl test in flapopt
