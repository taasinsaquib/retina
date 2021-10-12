#include <iostream>
#include <vector>
#include <assert.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>  // python interpreter
#include <pybind11/stl.h>    // type conversion

namespace py = pybind11;

int main() {

    py::scoped_interpreter guard{};

    // py::module_ sys = py::module_::import("sys");
    // py::print(sys.attr("path"));

    py::module_ sim = py::module_::import("simulation_func");

    py::object result = sim.attr("func")(5);
    int n = result.cast<int>();
    assert(n == 10);
}