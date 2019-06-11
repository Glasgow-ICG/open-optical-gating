// new (fails)

#ifndef __JPYTHONARRAY_H__
#define __JPYTHONARRAY_H__ 1

#define NEW_CODE 1

#include <Python.h>
#include "numpy/arrayobject.h"
#include <stdio.h>
#include <stdlib.h>
#include "jAssert.h"

// This should be specialized for all required types in JPythonArray.cpp
template<class Type> int ArrayType(void);

template<class Type> class JPythonArray
{
  protected:
#if NEW_CODE
	static const int kMaxDims = 3;
	npy_intp		dims[kMaxDims];
	npy_intp		strides[kMaxDims];
#else
	npy_intp		*dims;
	npy_intp		*strides;
#endif
	int				numDims;
	Type			*data;
//	PyArrayObject	*obj;		// I prefer not to store the object, because I think that's easier when dealing with sub-arrays.
								// It may be helpful to refcount it, though
	PyArrayObject	*mutableObj;		// I prefer not to store the object, because I think that's easier when dealing with sub-arrays.
										// It may be helpful to refcount it, though
	
	void AllocDims(int inNum, npy_intp *inDims, npy_intp *inStrides, int divideFactor = 1)
	{
#if NEW_CODE
		ALWAYS_ASSERT(inNum <= kMaxDims);
		numDims = inNum;
		memcpy(dims, inDims, sizeof(npy_intp) * numDims);
#else
		numDims = inNum;
		dims = new npy_intp[numDims];
		memcpy(dims, inDims, sizeof(npy_intp[numDims]));
		strides = new npy_intp[numDims];
#endif
		for (int i = 0; i < numDims; i++)
			strides[i] = inStrides[i] / divideFactor;
	}

	static void CheckArrayType(PyArrayObject *obj, int expectedDims = 0)
	{
		if (PyArray_TYPE(obj) != ArrayType())
		{
			// If this error is hit then the wrong array type was passed to the JPythonArray class
			PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Array type %d didn't match expected type %d", PyArray_TYPE(obj), ArrayType());
		}
		int dimCount = PyArray_NDIM(obj);
		if ((expectedDims != 0) && (dimCount != expectedDims))
		{
			// If this error is hit then an array with the wrong dimensions was passed to the JPythonArray class
			PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Array had the wrong number of dimensions (got %d, expected %d)\n", dimCount, expectedDims);
		}
	}	

  public:
	
	void Construct(PyArrayObject *obj, int expectedDims = 0)
	{
		CheckArrayType(obj, expectedDims);
		AllocDims(PyArray_NDIM(obj), PyArray_DIMS(obj), PyArray_STRIDES(obj), sizeof(Type));
		data = (Type *)PyArray_DATA(obj);
	}
	
	JPythonArray(PyArrayObject *obj, int expectedDims)
	{
		Construct(obj, expectedDims);
	}
	
	JPythonArray(PyObject *obj, int expectedDims)
	{
		if (PyArray_Check(obj))
			Construct((PyArrayObject *)obj, expectedDims);
		else
		{
			// If this error is hit then the wrong array type was passed to the JPythonArray class
			PyErr_Format(PyErr_NewException((char*)"exceptions.TypeError", NULL, NULL), "Object is not an array object");
		}
	}
	
	JPythonArray(Type *inData, int inNum, npy_intp *inDims, npy_intp *inStrides)
	{
		AllocDims(inNum, inDims, inStrides);
		data = inData;
	}
	
	virtual ~JPythonArray()
	{
#if NEW_CODE
#else
		delete[] dims;
		delete[] strides;
#endif
	}
	
	void SetZero(void)
	{
		// Set every element to zero (taking correct account of strides)
		int i;
		npy_intp indices[numDims];
		memset(indices, 0, sizeof(indices));
		do
		{
			npy_intp offset = 0;
			for (i = 0; i < numDims; i++)
				offset += strides[i] * indices[i];
			data[offset] = 0;
			for (i = numDims; i >= 0; i--)
			{
				indices[i]++;
				if (indices[i] == dims[i])
					indices[i] = 0;
				else
					break;
			}
		}
		while (i != -1);
		
/*		long numElements = 1;
		for (int i = 0; i < numDims; i++)
			numElements *= dims[i];
		memset(data, 0, sizeof(Type[numElements]));*/
	}

	npy_intp NumElements(void) const
	{
		npy_intp count = 1;
		for (int i = 0; i < numDims; i++)
			count *= dims[i];
		return count;
	}
	
	bool Contiguous(void) const
	{
		npy_intp expected = 1;
		for (int i = numDims - 1; i >= 0; i--)
		{
			if (strides[i] != expected)
				return false;
			expected *= dims[i];
		}
		return true;
	}
	
	void SetData(Type *inData, npy_intp len)
	{
		ALWAYS_ASSERT(len == NumElements());
		if (Contiguous())
		{
			memcpy(data, inData, len * sizeof(Type));
		}
		else
		{
			int i;
			npy_intp inPos = 0;
			npy_intp indices[numDims];
			memset(indices, 0, sizeof(indices));
			do
			{
				npy_intp offset = 0;
				for (i = 0; i < numDims; i++)
					offset += strides[i] * indices[i];
				data[offset] = inData[inPos++];
				for (i = numDims; i >= 0; i--)
				{
					indices[i]++;
					if (indices[i] == dims[i])
						indices[i] = 0;
					else
						break;
				}
			}
			while (i != -1);
		}
	}
	
	int NDims(void) const { return numDims; }
	npy_intp *Dims(void) { return dims; }		// This should be const, and return a const array, but PyArray_SimpleNew takes a non-const parameter for some reason
	npy_intp *Strides(void) { return strides; }	// This should be const, and return a const array, but PyArray_SimpleNew takes a non-const parameter for some reason
	Type *Data(void) const { return data; }
	static int ArrayType(void) { return ::ArrayType<Type>(); }
};

template<class Type> class JPythonArray1D : public JPythonArray<Type>
{
  public:
	JPythonArray1D(PyArrayObject *init) : JPythonArray<Type>(init, 1) { }
	JPythonArray1D(PyObject *init) : JPythonArray<Type>(init, 1) { }
	JPythonArray1D(Type *inData, npy_intp *inDims, npy_intp *inStrides) : JPythonArray<Type>(inData, 1, inDims, inStrides) { }

	Type &operator[](int i)		// Note we return a reference here, so that this can be used as an lvalue, e.g. my1DArray[0] = 1.0, or my2DArray[0][0] = 1.0;
	{
//		printf("Access element %d of %d\n", i, JPythonArray<Type>::dims[0]);
		ALWAYS_ASSERT(i < JPythonArray<Type>::dims[0]);
		return JPythonArray<Type>::data[i * JPythonArray<Type>::strides[0]];
	}
	
	Type &GetIndex_CanPromote(int i)
	{
		// Behaves like operator[], but if we have a single value in the array then returns that value regardless of i
		// This isn't ideal - it's a way of working around the fact that the object used to initialize this array may be a scalar value
		if (JPythonArray<Type>::dims[0] == 1)
			return JPythonArray<Type>::data[0];
		else
			return operator[](i);
	}
};

template<class Type> class JPythonArray2D : public JPythonArray<Type>
{
  public:
	JPythonArray2D(PyArrayObject *init) : JPythonArray<Type>(init, 2) { }
	JPythonArray2D(PyObject *init) : JPythonArray<Type>(init, 2) { }
	JPythonArray2D(Type *inData, npy_intp *inDims, npy_intp *inStrides) : JPythonArray<Type>(inData, 2, inDims, inStrides) { }

	JPythonArray1D<Type> operator[](int i)
	{
		// Could check that i is in range (check against dims[0])
		return JPythonArray1D<Type>(JPythonArray<Type>::data + JPythonArray<Type>::strides[0] * i, JPythonArray<Type>::dims + 1, JPythonArray<Type>::strides + 1);
	}
};

template<class Type> class JPythonArray3D : public JPythonArray<Type>
{
  public:
	JPythonArray3D(PyArrayObject *init) : JPythonArray<Type>(init, 3) { }
	JPythonArray3D(PyObject *init) : JPythonArray<Type>(init, 3) { }

	JPythonArray2D<Type> operator[](int i)
	{
		// Could check that i is in range (check against dims[0])
		return JPythonArray2D<Type>(JPythonArray<Type>::data + JPythonArray<Type>::strides[0] * i, JPythonArray<Type>::dims + 1, JPythonArray<Type>::strides + 1);
	}
};

template<class Type> JPythonArray2D<Type> PromoteTo2D(PyArrayObject *init)
{
	if (PyArray_NDIM(init) == 1)
	{
		npy_intp dims[2] = { 1, PyArray_DIMS(init)[0] };
		npy_intp strides[2] = { 0, PyArray_STRIDES(init)[0] / sizeof(Type) };
		return JPythonArray2D<Type>((Type *)PyArray_DATA(init), dims, strides);
	}
	else
	{
		// This could fail (if for example we are given a 3D array), but if that happens then a suitable error should be reported
		return JPythonArray2D<Type>(init);
	}
}

#endif
