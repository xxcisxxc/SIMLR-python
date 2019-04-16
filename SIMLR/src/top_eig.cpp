#include <stdio.h>
#include <math.h>
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <SymEigsSolver.h>
#include <MatOp/SparseSymMatProd.h>
#include <time.h>
#include <Python.h>
#include <numpy/arrayobject.h>
using namespace Eigen;
using namespace Spectra;


void make_top_eigenvectors(double *val,double *ind, int KK, int NN, int NK, double *eigenvectors,double *eigenvalues){
    clock_t begin = clock();
    Eigen::SparseMatrix<double> mat((const int) NN,(const int) NN);         // default is column major
    mat.reserve(Eigen::VectorXi::Constant((const int) NN, (const int) NK));
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve((const int) NN*NK);
    for(int i=0; i<NN; i++){
        for (int j = 0; j< NK; j++){
            tripletList.push_back(T((int) ind[i*NK+j],i,val[i*NK+j]));
            //std::cout << ind[i*NK+j] << std::endl << val[i*NK+j] << std::endl;
        }
    }
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    mat += Eigen::SparseMatrix<double>(mat.transpose());
    clock_t end = clock();
    //printf("Elapsed time in initialization is %f seconds\n", (double)(end - begin)/CLOCKS_PER_SEC);
    SparseSymMatProd<double> op(mat);
    begin = clock();
    // Construct eigen solver object, requesting the largest KK eigenvalues
    SymEigsSolver< double, LARGEST_ALGE, SparseSymMatProd<double> > eigs(&op, KK, 2*KK);
    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();
    // Retrieve results
    
    Eigen::VectorXd evalues;
    Eigen::MatrixXd evectors;
    if(eigs.info() == SUCCESSFUL){
        evalues = eigs.eigenvalues();
        evectors = eigs.eigenvectors();
    }
    //std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    end = clock();
    //printf("Elapsed time in eigen-decomposition is %f seconds\n", (double)(end - begin)/CLOCKS_PER_SEC);
    ///
    begin = clock();
    for (int j = 0; j< KK; j++){
        eigenvalues[j] = evalues[j];
        for (int i = 0; i < NN; i++){
            eigenvectors[i*KK+j] = evectors.col(j)[i];
        }
    }
   end = clock();
    //printf("Elapsed time in copying eigenvectors is %f seconds\n", (double)(end - begin)/CLOCKS_PER_SEC);
    ///
}

PyObject * top_eig(PyObject *in0, PyObject *in1, PyObject *scal){
//PyObject * top_eig(PyArrayObject *in0, PyArrayObject *in1, PyObject *scal){
    //PyArrayObject *in0_arr = NULL;
    PyArrayObject *in0_arr = (PyArrayObject *)in0;
    //in0_arr = PyArray_GETCONTIGUOUS((PyArrayObject *)in0);
    //in0_arr = PyArray_GETCONTIGUOUS(in0);

    double *val = NULL;
    val = (double *)PyArray_DATA(in0_arr);

    //PyArrayObject *in1_arr = NULL;
    PyArrayObject *in1_arr = (PyArrayObject *)in1;
    //in1_arr = PyArray_GETCONTIGUOUS((PyArrayObject *)in1);
    //in1_arr = PyArray_GETCONTIGUOUS(in1);

    double *ind = NULL;
    ind = (double *)PyArray_DATA(in1_arr);
    
    int KK = (int)PyLong_AsLong(scal);

    int NN, NK;
    npy_intp *dims = NULL;
    dims = PyArray_DIMS(in0_arr);
    NN = dims[0];

    dims = PyArray_DIMS(in1_arr);
    NK = dims[1];

    dims[0] = (npy_intp)NN;
    dims[1] = (npy_intp)KK;

    PyObject *out1 = NULL;
    out1 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    PyArrayObject *out1_arr = NULL;
    out1_arr = PyArray_GETCONTIGUOUS((PyArrayObject *)out1);

    double *eigenvectors = NULL;
    eigenvectors = (double *)PyArray_DATA(out1_arr);

    dims[0] = dims[1];
    int tmp = 1;
    dims[1] = (npy_intp)tmp;

    PyObject *out2 = NULL;
    out2 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    PyArrayObject *out2_arr = NULL;
    out2_arr = PyArray_GETCONTIGUOUS((PyArrayObject *)out2);

    double *eigenvalues = NULL;
    eigenvalues = (double *)PyArray_DATA(out2_arr);

    make_top_eigenvectors(val, ind, KK, NN, NK, eigenvectors, eigenvalues);

    PyObject *outs;
    outs = PyTuple_New(2);
    PyTuple_SetItem(outs, 0, (PyObject *)out1_arr);
    PyTuple_SetItem(outs, 1, (PyObject *)out2_arr);
    return outs;
}

static PyObject * _top_eig(PyObject *self, PyObject *args){
    PyObject *in0, *in1;
    //PyArrayObject *in0, *in1;
    PyObject *scal = NULL;
    /*if (!PyArg_ParseTuple(args, "O!O!O", &PyArray_Type, &in0, &PyArray_Type, &in1, &scal)){
        return NULL;
    }*/
    if (!PyArg_ParseTuple(args, "OOO", &in0, &in1, &scal)){
        return NULL;
    }

    in0 = PyArray_FROM_OTF(in0, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    in1 = PyArray_FROM_OTF(in1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *out = NULL;
    out = top_eig(in0, in1, scal);
    return out;
}

static PyMethodDef topeigMethods[] = {
    {
        "top_eig", 
        _top_eig, 
        METH_VARARGS, 
        ""
    }, 
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef topeigModule = {
    PyModuleDef_HEAD_INIT, 
    "topeigModule", 
    NULL, 
    -1, 
    topeigMethods
};

PyMODINIT_FUNC PyInit_top_eig(void){
    import_array();
    return PyModule_Create(&topeigModule);
}