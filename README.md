# retina
Re-creating a retina model with Open3D  


## Calling Python from C++
* use pybind11 (easier than using Python API)
* install: https://pybind11.readthedocs.io/en/latest/installing.html
* https://pybind11.readthedocs.io/en/latest/installing.html?highlight=environment#include-with-pypi


### CMAKE
* install: https://cmake.org/install/
* hello world: https://tuannguyen68.gitbooks.io/learning-cmake-a-beginner-s-guide/content/chap1/chap1.html
* Compile pybind11: https://pybind11.readthedocs.io/en/stable/advanced/embedding.html
    * `cmake -H. -Bbuild`
    * `cmake --build build -- -j3`