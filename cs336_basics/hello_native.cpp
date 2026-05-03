#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* hello(PyObject* self, PyObject* args) {
    return PyUnicode_FromString("hello from c++");
}

static PyMethodDef hello_methods[] = {
    {"hello", hello, METH_NOARGS, "Return a hello-world string from C++."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef hello_module = {
    PyModuleDef_HEAD_INIT,
    "_hello",
    "Minimal C++ hello-world extension.",
    -1,
    hello_methods,
};

PyMODINIT_FUNC PyInit__hello(void) {
    return PyModule_Create(&hello_module);
}
