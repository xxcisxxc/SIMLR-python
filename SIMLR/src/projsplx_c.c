#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

PyObject * projsplx_c(PyObject *in){
	//int numDims, m, n, k, d, j, npos, ft;
	//double *s, *x, *vs;
	//double sumResult = -1, tmpValue, tmax, f,lambda_m;

	//PyArrayObject *in_arr;
	//if !PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_arr){
	//	return NULL
	//}
	PyArrayObject *in_arr = NULL;
	in_arr = PyArray_GETCONTIGUOUS(in);

	double *y = NULL;
	y = (double *)PyArray_DATA(in_arr);

	int numDims = 0;
	numDims = PyArray_NDIM(in_arr);

	npy_intp *dims = NULL;
	dims = PyArray_DIMS(in_arr);

	int m, n;
	m = dims[0];
	n = dims[1];

	PyObject *out = NULL;
	out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

	PyArrayObject *out_arr = NULL;
	out_arr = PyArray_GETCONTIGUOUS(out);

	double *x = NULL;
	x = (double *)PyArray_DATA(out_arr);

	double *s, *vs;
	s = (double *)calloc(m, sizeof(double));
	vs = (double *)calloc(m, sizeof(double));

	int k, j, ft, npos;
	double f, lambda_m;

	for (k = 0; k < n; k++){
		double means = 0;
		double mins = 100000;
		for (j = 0; j < m; j++){
			//s[j] = y[j+k*m];
			s[j] = y[j*n+k];
			//npy_intp ind[2] = NULL;
			//ind[0] = j; ind[1] = k;
			//s[j] = *PyArray_GetPtr(in_arr, ind);
			means += s[j];
			mins = (mins > s[j]) ? s[j] : mins;
		}

		for (j = 0; j < m; j++){
			s[j] -= (means - 1) / m;
		}
		ft = 1;
		if (mins < 0){
			f = 1;
			lambda_m = 0;
			while (fabs(f) > 1e-10){
				npos = 0;
				f = 0;
				for (j = 0; j < m; j++){
					vs[j] = s[j] - lambda_m;
					if (vs[j] > 0){
						npos += 1;
						f += vs[j];
					}
				}
				lambda_m += (f - 1) / npos;
				if (ft > 100){
					for (j = 0; j < m; j++){
						x[j*n+k] = (vs[j] > 0) ? vs[j] : 0;
					}
					break;
				}
				ft += 1;
			}
			for (j = 0; j < m; j++){
				x[j*n+k] = (vs[j] > 0) ? vs[j] : 0;
			}
		}
		else{
			for (j = 0; j < m; j++){
				x[j*n+k] = s[j];
			}
		}
	}
	return out_arr;
}

static PyObject * _projsplx_c(PyObject *self, PyObject *args){
	PyObject *in = NULL;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in)){
		return NULL;
	}

	PyObject *out = NULL;
	out = projsplx_c(in);
	return out;
}

static PyMethodDef ProjsplxMethods[] = {
	{
		"projsplx_c", 
		_projsplx_c, 
		METH_VARARGS, 
		""
	}, 
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef ProjsplxModule = {
	PyModuleDef_HEAD_INIT, 
	"ProjsplxModule", 
	NULL, 
	-1, 
	ProjsplxMethods
};

PyMODINIT_FUNC PyInit_projsplx_c(void){
	import_array();
	return PyModule_Create(&ProjsplxModule);
}