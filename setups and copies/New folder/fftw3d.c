#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fftw_helper.h"
#include <string.h>

//////////////////////
// C FFT functions
//////////////////////

void fft3d_c2c(fftw_complex *data, int Nx, int Ny, int Nz, int direction) {
    fftw_plan plan = fftw_plan_dft_3d(Nx, Ny, Nz, data, data, direction, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

void fft3d_r2c(double *in, fftw_complex *out, int Nx, int Ny, int Nz) {
    fftw_plan plan = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

void fft3d_c2r(fftw_complex *in, double *out, int Nx, int Ny, int Nz) {
    fftw_plan plan = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

//////////////////////
// Python wrapper
//////////////////////

static PyObject* py_fft3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject *arr_obj, *out_obj = NULL;
    const char *mode = "c2c";
    const char *direction_str = "forward";

    static char *kwlist[] = {"arr", "mode", "direction", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|ssO!", kwlist,
                                     &PyArray_Type, &arr_obj,
                                     &mode, &direction_str,
                                     &PyArray_Type, &out_obj))
        return NULL;

    int direction = (strcmp(direction_str, "forward") == 0) ? FFTW_FORWARD : FFTW_BACKWARD;

    int Nx = (int)PyArray_DIM(arr_obj, 0);
    int Ny = (int)PyArray_DIM(arr_obj, 1);
    int Nz = (int)PyArray_DIM(arr_obj, 2);

    if (strcmp(mode, "c2c") == 0) {
        PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)arr_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY);
        if (!arr) return NULL;

        fft3d_c2c((fftw_complex*)PyArray_DATA(arr), Nx, Ny, Nz, direction);

        PyArray_ResolveWritebackIfCopy(arr);
        return PyArray_Return(arr);
    }
    else if (strcmp(mode, "r2c") == 0) {
        PyArrayObject *arr_in = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)arr_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (!arr_in) return NULL;

        int out_Nz = Nz/2 + 1;
        npy_intp dims[3] = {Nx, Ny, out_Nz};
        PyArrayObject *arr_out = out_obj ? (PyArrayObject*)PyArray_FROM_OTF((PyObject*)out_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY)
                                         : (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_COMPLEX128);

        fft3d_r2c((double*)PyArray_DATA(arr_in), (fftw_complex*)PyArray_DATA(arr_out), Nx, Ny, Nz);

        Py_DECREF(arr_in);
        return PyArray_Return(arr_out);
    }
    else if (strcmp(mode, "c2r") == 0) {
        PyArrayObject *arr_in = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)arr_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        if (!arr_in) return NULL;

        PyArrayObject *arr_out = out_obj ? (PyArrayObject*)PyArray_FROM_OTF((PyObject*)out_obj, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY)
                                         : (PyArrayObject*)PyArray_SimpleNew(3, PyArray_DIMS(arr_obj), NPY_FLOAT64);

        fft3d_c2r((fftw_complex*)PyArray_DATA(arr_in), (double*)PyArray_DATA(arr_out), Nx, Ny, Nz);

        Py_DECREF(arr_in);
        return PyArray_Return(arr_out);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Invalid mode: must be 'c2c', 'r2c', or 'c2r'");
        return NULL;
    }
}

//////////////////////
// Module definition
//////////////////////

static PyMethodDef FFTW3DMethods[] = {
    {"fft3d", (PyCFunction)py_fft3d, METH_VARARGS | METH_KEYWORDS,
     "Perform 3D FFT in-place. Modes: 'c2c', 'r2c', 'c2r'. Optional 'direction'='forward'/'backward'. Optional 'out' for r2c/c2r."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fftw3dmodule = {
    PyModuleDef_HEAD_INIT,
    "fftw3d",
    "3D FFT module using FFTW",
    -1,
    FFTW3DMethods
};

PyMODINIT_FUNC PyInit_fftw3d(void)
{
    import_array();
    return PyModule_Create(&fftw3dmodule);
}
