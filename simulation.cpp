#include <iostream>
#include <vector>
#include <assert.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>  // python interpreter
#include <pybind11/stl.h>    // type conversion

namespace py = pybind11;

int main() {

    // python interpreter now lives in this scope
    py::scoped_interpreter guard{};

    // TODO: check if vector is being copied; memory usage
    std::vector<double> onv(14*3, 1);   // create ONV

    // call the function
    py::module_ sim = py::module_::import("simulation_func");
    py::object result = sim.attr("func")(py::cast(onv));

    // use the output
    std::vector<double> n = result.cast<std::vector<double>>();
    for (int i = 0; i < 2; i++)
        std::cout << n[i] << " ";
    std::cout << std::endl;
}
