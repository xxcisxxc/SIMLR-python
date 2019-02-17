#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

PyObject * Kbeta(PyObject *in0, PyObject *in1, int *in2=NULL){
	PyArrayObject *in0_arr = PyArray_GETCONTIGUOUS((PyArrayObject *)in0);
	PyArrayObject *in1_arr = PyArray_GETCONTIGUOUS((PyArrayObject *)in1);

	if (PyArray_NDIM(in1_arr) != 2){
		PyErr_SetString(PyExc_ValueError, "Second input argument must be 2D");
		return NULL;
	}

	unsigned int nDims = PyArray_NDIM(in0_arr);
	const npy_intp *dims = PyArray_DIMS(in0_arr);

	double *K = (double *)PyArray_DATA(in0_arr);
	double *beta = (double *)PyArray_DATA(in1_arr);

	unsigned int n1, n2, m;
	if (nDims == 2){
		n1 = dims[0];
		n2 = dims[1];
		m = 1;
	}
	else if (nDims == 3){
		n1 = dims[1];
		n2 = dims[2];
		m = dims[0];
	}
	else{
		PyErr_SetString(PyExc_ValueError, "First input arg must be 2D or 3D");
		return NULL;
	}
	unsigned int nn = n1 * n2;

	if (PyArray_DIMS(in1_arr)[0] != m){
		PyErr_SetString(PyExc_ValueError, "Input dimensions mismatch");
		return NULL;
	}

	bool symmetric = false;
	if ((n1 == n2) && (in2 != NULL)){
		symmetric = (bool) *in2;
	}

	npy_intp nd[2] = {n1, n2};
	PyObject *out = PyArray_SimpleNew(2, nd, NPY_DOUBLE);
	PyArrayObject *out_arr = PyArray_GETCONTIGUOUS((PyArrayObject *)out);
	double *Kbeta_ = (double *)PyArray_DATA(out_arr);

	if (symmetric){
		for (unsigned int k = 0; k < m; k++){
			if (beta[k] <= 1e-8){
				continue;
			}
			for (unsigned int i = 0; i < n2; i++){
				for (unsigned int j = i; j < n1; j++){
					//Kbeta_[i*n1+j] += beta[k] * K[k*nn+i*n1+j];
					Kbeta_[i+j*n2] += beta[k] * K[k*nn+i+j*n2];
				}
			}
		}
		for (unsigned int i = 0; i < n2; i++){
			for (unsigned int j = i+1; j < n1; j++){
				Kbeta_[j+i*n2] = Kbeta_[i+j*n2];
			}
		}
	}
	else{
		for (unsigned int k = 0; k < m; k++){
			if (beta[k] <= 1e-8){
				continue;
			}
			for (unsigned int i = 0; i < n2; i++){
				for (unsigned int j = 0; j < n1; j++){
					Kbeta_[i+j*n2] += beta[k] * K[k*nn+i+j*n2];
				}
			}
		}
	}
	return (PyObject *)out_arr;
}

static PyObject * _Kbeta(PyObject *self, PyObject *args){
	PyObject *in0, *in1;
	int *in2=NULL;
	int args_size = (int)PyTuple_Size(args);
	if (args_size == 2){
		if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &in0, &PyArray_Type, &in1)){
			return NULL;
		}
	}
	else if (args_size == 3){
		if (!PyArg_ParseTuple(args, "O!O!p", &PyArray_Type, &in0, &PyArray_Type, &in1, &in2)){
			return NULL;
		}
	}

	PyObject *out = NULL;
	out = Kbeta(in0, in1, in2);
	return out;
}

static PyMethodDef KbetaMethods[] = {
	{
		"Kbeta", 
		_Kbeta, 
		METH_VARARGS, 
		""
	}, 
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef KbetaModule = {
	PyModuleDef_HEAD_INIT, 
	"KbetaModule", 
	NULL, 
	-1, 
	KbetaMethods
};

PyMODINIT_FUNC PyInit_projsplx_c(void){
	import_array();
	return PyModule_Create(&KbetaModule);
}