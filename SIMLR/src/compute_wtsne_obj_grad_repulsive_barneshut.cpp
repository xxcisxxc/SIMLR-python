#include <math.h>
#include "barnes_hut.h"
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>

void getRepulsiveObjGradi(int id, double *pos, QuadTree* tree, double farness_factor, int nArgOut, double *pqsum, double *grad) {
    double dist2, dist, diff, decoef, q;
    int i,d;
    double tmp;
    
    if (tree==NULL) return;
    if (tree->node!=NULL && tree->node->id==id) return;
    
    dist2 = 0.0;
    for (d=0;d<2;d++) {
        diff = pos[d]-tree->position[d];
        dist2 += diff*diff;
    }
    
    tmp = farness_factor*getTreeWidth(tree);
    if (tree->childCount>0 && dist2<tmp*tmp) {
        for(i=0;i<tree->childrenLength;i++) {
            if (tree->children[i]!=NULL) {
                getRepulsiveObjGradi(id, pos, tree->children[i], farness_factor, nArgOut, pqsum, grad);
            }
        }
    } else {
        q = 1.0 / (1+dist2);
        (*pqsum) += q * tree->weight;
        
        if (nArgOut>1) {
            tmp = tree->weight * q * q;
            for (d=0;d<2;d++) {
                grad[d] += (tree->position[d] - pos[d]) * tmp;
            }
        }
    }
}

void getRepulsiveObjGrad(double *Y, double *weights, int n, double eps, double farness_factor, int nArgOut, double *pobj, double *grad) {
    double pos[2], gradi[2], d2, diff;
    double *coef;
    int i, j, d;
    QuadTree *tree;
    double qsumi, qsum;
    
    tree = buildQuadTree(Y,weights,n);

    qsum = 0.0;
    for (i=0;i<n;i++) {
        for(d=0;d<2;d++) {
            gradi[d] = 0.0;
            pos[d] = Y[d+i*2];
        }
        qsumi = 0.0;
        getRepulsiveObjGradi(i, pos, tree, farness_factor, nArgOut, &qsumi, gradi);
        qsum += qsumi*weights[i];
        if (nArgOut>1) {
            for(d=0;d<2;d++)
                grad[d+i*2] = 4 * gradi[d] * weights[i];
        }
    }
    
    if (nArgOut>1) {
        for (i=0;i<n;i++) {
            for(d=0;d<2;d++)
                grad[d+i*2] /= qsum;
        }
    }

    (*pobj) = log(qsum+eps);
    
    destroyQuadTree(tree);
}

/*
Pyobject * compute_wtsne_obj_grad_repulsive_barneshut(){
	double *Y = NULL;

}
*/
PyObject * compute_wtsne_obj_grad_repulsive_barneshut(PyObject *in0, PyObject *in1, PyObject *scal0, PyObject *scal1){
    PyArrayObject *in0_arr = (PyArrayObject *)in0;

    double *Y = NULL;
    Y = (double *)PyArray_DATA(in0_arr);

    PyArrayObject *in1_arr = (PyArrayObject *)in1;

    double *weights = NULL;
    weights = (double *)PyArray_DATA(in1_arr);
    
    double farness_factor = PyFloat_AsDouble(scal0);
    int nArgOut = PyLong_AsLong(scal1);

    npy_intp *dims = NULL;
    dims = PyArray_DIMS(in0_arr);
    int n = dims[0];

    double eps = 2.220446049250313e-16;

    int tmp = 2;
    dims[1] = (npy_intp)tmp;

    double *pobj = NULL;
    double temp = 0.0;
    pobj = &temp;

    double *grad = NULL;
    PyArrayObject *out1_arr = NULL;


    if (nArgOut>1) {
    	PyObject *out1 = NULL;
    	out1 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    	out1_arr = PyArray_GETCONTIGUOUS((PyArrayObject *)out1);

    	grad = (double *)PyArray_DATA(out1_arr);
    }

    getRepulsiveObjGrad(Y, weights, n, eps, farness_factor, nArgOut, pobj, grad);

    PyObject *outs;
    outs = PyTuple_New(2);
    PyTuple_SetItem(outs, 0, PyFloat_FromDouble(*pobj));
    PyTuple_SetItem(outs, 1, (PyObject *)out1_arr);
    return outs;
}

static PyObject * _compute_wtsne_obj_grad_repulsive_barneshut(PyObject *self, PyObject *args){
    PyObject *in0, *in1;
    PyObject *scal0, *scal1 = NULL;

    if (!PyArg_ParseTuple(args, "OOOO", &in0, &in1, &scal0, &scal1)){
        return NULL;
    }

    in0 = PyArray_FROM_OTF(in0, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    in1 = PyArray_FROM_OTF(in1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *out = NULL;
    out = compute_wtsne_obj_grad_repulsive_barneshut(in0, in1, scal0, scal1);
    return out;
}

static PyMethodDef computewtsneobjgradrepulsivebarneshutMethods[] = {
    {
        "compute_wtsne_obj_grad_repulsive_barneshut", 
        _compute_wtsne_obj_grad_repulsive_barneshut, 
        METH_VARARGS, 
        ""
    }, 
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef computewtsneobjgradrepulsivebarneshutModule = {
    PyModuleDef_HEAD_INIT, 
    "computewtsneobjgradrepulsivebarneshutModule", 
    NULL, 
    -1, 
    computewtsneobjgradrepulsivebarneshutMethods
};

PyMODINIT_FUNC PyInit_compute_wtsne_obj_grad_repulsive_barneshut(void){
    import_array();
    return PyModule_Create(&computewtsneobjgradrepulsivebarneshutModule);
}