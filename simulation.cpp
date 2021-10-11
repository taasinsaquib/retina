// #include <Python/Python.h>
#include "Python.h"     // squiggle goes away w/ compile flag

#include <stdio.h>

int main() {
     
    // boilerplate
    Py_Initialize();
    PyObject* sys = PyImport_ImportModule("sys");
    PyObject* path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyString_FromString("."));
     
    PyObject* ModuleString = PyString_FromString((char*) "simulation_func");
    PyObject* Module = PyImport_Import(ModuleString);
    PyObject* Dict = PyModule_GetDict(Module);
    
    PyObject* Func = PyDict_GetItemString(Dict, "func");
    PyObject* args = PyTuple_Pack(1, PyFloat_FromDouble(2.0));
    PyObject* Result = PyObject_CallObject(Func, args);

//    Py_DECREF(pValue);
//    Clean up
//    Py_DECREF(pModule);
//    Py_DECREF(pName);

    Py_Finalize();
}