#define PY_ARRAY_UNIQUE_SYMBOL j_pyarray
#include "Python.h"
#include "numpy/arrayobject.h"
#include "../../JAssert.h"
#include "../../JPythonCommon.h"
#include "../../jCoord.h"
#include "../../JMovieReader.h"
#include <vector>
#include <algorithm>

extern "C" PyObject *create_reader(PyObject *self, PyObject *args)
{
	// inputs
	int x, y;
	const char *path;
	// outputs

	// parse the input arrays from *args
	if (!PyArg_ParseTuple(args, "(dd)s", 
							&x, &y, &path))
	{
		PyErr_Format(PyErr_NewException("exceptions.TypeError", NULL, NULL), "Unable to parse arguments to create_reader!");
		return NULL;
	}
	
	Point windowPos = { y, x };
	JMovieReader *result = new JMovieReader(windowPos, path);
		
	return (PyObject *)result;		//??????
}

/* Define a methods table for the module */

static PyMethodDef jmovies_methods[] = {
	{"create_reader", create_reader, METH_VARARGS},
	{NULL,NULL} };



/* initialisation - register the methods with the Python interpreter */

extern "C" void initjmovies(void)
{
	(void) Py_InitModule("jmovies", jmovies_methods);
	import_array();
}
