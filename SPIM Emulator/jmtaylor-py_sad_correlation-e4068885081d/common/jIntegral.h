/*	Template: jIntegral.h

	Copyright 2010-2015 Jonathan Taylor. All rights reserved.
 
	Template classes for evaluating integrals numerically.
	This is intended to hide the detail of a numerical integral as much as possible,
	while maintaining the same performance when compiler optimizations are enabled.
	
	The code will split the integration across multiple threads.
	We assume that the cost of evaluating each sample point is high.
	If that is not the case, the code will not perform well
	(we will be swamped with inter-thread synchronization).
	If that is a problem then we should look into adding an option to disable threading.
	
	Usage:
		Global declaration:
			DEFINE_CLASS_FUNCTOR([return type], [class name], [point evaluation method], [functor type name])

		Code:
			[functor type name] functor(this);
			Integrate<[return type], [kernel type]>(functor, 0, PI, 100);
*/

#ifndef __J_INTEGRAL_H__
#define __J_INTEGRAL_H__ 1

#include <vector>
#if USE_WORK_THREADS_LITE
	#include "WorkThreads.h"
#else
	#include "WorkThreads2.h"
#endif
#include "ObjectPool.h"

inline double SimpsonsMultiplier(int i, int i_max)
{
	if (i == 0 || i == i_max)
		return 1.0 / 3.0;
	else if (i & 1)
		return 4.0 / 3.0;
	return 2.0 / 3.0;
}

inline double SimpsonsMultiplier(double x, double x_0, double x_1, double dx)
{
	int i = (int)((x - x_0 + 0.1*dx) / dx);
	int i_max = (int)((x_1 - x_0 + 0.1*dx) / dx);
	return SimpsonsMultiplier(i, i_max);
}

inline double TrapeziumMultiplier(int i, int i_max)
{
	if (i == 0 || i == i_max)
		return 1.0 / 2.0;
	return 1.0;
}

namespace jIntegralPrivate
{
	// Implementation-specific code. Code should generally call Integrate() and friends instead of
	// delving into the details in this namespace
	
	template<class ResultType> ResultType GetZeroSum(void);
	template<class ResultType> ResultType SumResultList(std::vector<ResultType> &resultList);

	template<class ResultType, class Eval, bool moreAccuracy> struct CallbackParms
	{
		ObjectPool<ResultType>	threadResults;
		ObjectPool<std::vector<ResultType> > threadResultLists;
		const Eval	*eval;
		
		CallbackParms(const Eval *inEval) : threadResults(kAbsoluteMaxThreads), threadResultLists(kAbsoluteMaxThreads), eval(inEval) { }
		
		void AddResult(ResultType val)
		{
			bool firstTime;
			if (moreAccuracy)
			{
				std::vector<ResultType> *temp = threadResultLists.GetAvailableObject(&firstTime);
				if (firstTime)
					temp->erase(temp->begin(), temp->end());
				temp->push_back(val);
				threadResultLists.FinishedWithObject(temp);
			}
			else
			{
				ResultType *temp = threadResults.GetAvailableObject(&firstTime);
				if (firstTime)
					*temp = GetZeroSum<ResultType>();
				*temp += val;
				threadResults.FinishedWithObject(temp);
			}
		}
	};

	template<class ResultType, class Eval, class Kernel, bool moreAccuracy> class Integral
	{
		/*	This class performs a 1D integral (potentially using multiple threads to divide the work).
			It is possible to specify increased accuracy (for considerably reduced performance) by setting
			the moreAccuracy flags.
			If this is set then the same values are summed in ascending order of their magnitude.
			That guards against roundoff error when summing huge numbers of values.	
			It should not be necessary to use this class directly - the functions at the end of this
			file such as Integrate, IntegrateOverCircle should generally be used	*/
	  protected:
		double x_0, x_1, dx;
		int xIntervals;
	  
		typedef Integral<ResultType, Eval, Kernel, moreAccuracy> OurType;
		typedef CallbackParms<ResultType, Eval, moreAccuracy> CBParmsType;

		ResultType EvaluateIntegrandForI(const Eval *eval, int i, double outerOuterVal, double outerVal) const
		{
			double x = x_0 + i * dx;
			return eval->Evaluate(outerOuterVal, outerVal, x) * Kernel::Evaluate(x) * SimpsonsMultiplier(i, xIntervals) * dx;
		}
		
		ResultType EvaluateInnerIntegralUnthreaded(const Eval *eval, double outerOuterVal, double outerVal) const
		{
			ResultType sum = GetZeroSum<ResultType>();
			if (moreAccuracy)
			{
				std::vector<ResultType> results;
				for (int i = 0; i <= xIntervals; i++)
					results.push_back(EvaluateIntegrandForI(eval, i, outerOuterVal, outerVal));
				sum = SumResultList(results);
			}
			else
				for (int i = 0; i <= xIntervals; i++)
					sum += EvaluateIntegrandForI(eval, i, outerOuterVal, outerVal);
			return sum;
		}

		virtual void IntegerWorkCallback(int thisItem, void *userData)
		{
			// This version will be called in the case of a single integral
			// Integral2D::IntegerWorkCallback will be called for a double integral
			CBParmsType *cbParms = (CBParmsType *)userData;
			cbParms->AddResult(EvaluateIntegrandForI(cbParms->eval, thisItem, 0.0, 0.0));
		}
		
		DEFINE_INTEGER_WORK_FUNCTOR(OurType, IntegerWorkCallback, CallbackFunctor, callbackFunctor, CBParmsType);		

		ResultType InternalEvaluate(const Eval *eval, int intervals)
		{
			CBParmsType	cbParms(eval);
			gWorkThreads->DoWorkWithFunctor(&callbackFunctor, 0, intervals, 1, &cbParms);

			ResultType result = GetZeroSum<ResultType>();
			if (moreAccuracy)
			{
				std::vector<ResultType> results;
				for (int i = 0; i < cbParms.threadResultLists.NumObjects(); i++)
					results.push_back(SumResultList(*cbParms.threadResultLists.GetIndObject(i)));
				result = SumResultList(results);
			}
			else
				cbParms.threadResults.MergeIntoDestination(&result);
			
			return result;
		}

	  public:
		Integral(int inxIntervals) { }
		Integral(double inx_0, double inx_1, int inxIntervals) : callbackFunctor(this)
		{
			x_0 = inx_0;
			x_1 = inx_1;
			xIntervals = inxIntervals;
			dx = (x_1 - x_0) / xIntervals;
		
			// Simpson's rule requires an even number of intervals (= odd number of sample points)
			ALWAYS_ASSERT(!(xIntervals & 1));
		}
		
		virtual ~Integral() { }

		virtual ResultType Evaluate(const Eval *eval)
		{
			return InternalEvaluate(eval, xIntervals);
		}
	};

	template<class ResultType, class InnerEval, class KernelX, class KernelY, bool moreAccuracy> class Integral2D : public Integral<ResultType, InnerEval, KernelY, moreAccuracy>
	{
		/*	This class performs a 2D integral (potentially using multiple threads to divide the work). 
			It should not be necessary to use this class directly - the functions at the end of this
			file such as Integrate, IntegrateOverCircle should generally be used	*/
	  protected:
		double x_0, x_1, dx;
		int xIntervals;
  		typedef CallbackParms<ResultType, InnerEval, moreAccuracy> CBParmsType;

		ResultType EvaluateMiddleIntegralUnthreaded(const InnerEval *eval, double outerVal) const
		{
			ResultType sum = GetZeroSum<ResultType>();

			if (moreAccuracy)
			{
				std::vector<ResultType> results;
				for (int i = 0; i <= xIntervals; i++)
				{
					double x = i * dx;
					results.push_back(Integral<ResultType, InnerEval, KernelY, moreAccuracy>::EvaluateInnerIntegralUnthreaded(eval, outerVal, x) * KernelX::Evaluate(x) * SimpsonsMultiplier(i, xIntervals) * dx);
				}
				sum = SumResultList(results);
			}
			else
				for (int i = 0; i <= xIntervals; i++)
				{
					double x = i * dx;
					sum += Integral<ResultType, InnerEval, KernelY, moreAccuracy>::EvaluateInnerIntegralUnthreaded(eval, outerVal, x) * KernelX::Evaluate(x) * SimpsonsMultiplier(i, xIntervals) * dx;
				}


			return sum;
		}

		virtual void IntegerWorkCallback(int thisItem, void *userData)
		{
			// This version will be called in the case of a double integral
			// Integral3D::IntegerWorkCallback will be called for a triple integral
			CBParmsType *cbParms = (CBParmsType *)userData;
			
			double x = x_0 + thisItem * dx;
			cbParms->AddResult(Integral<ResultType, InnerEval, KernelY, moreAccuracy>::EvaluateInnerIntegralUnthreaded(cbParms->eval, 0.0, x) * KernelX::Evaluate(x) * SimpsonsMultiplier(thisItem, xIntervals) * dx);
		}

	  public:
		Integral2D(double inx_0, double inx_1, int inxIntervals, double y_0, double y_1, int yIntervals) : Integral<ResultType, InnerEval, KernelY, moreAccuracy>(y_0, y_1, yIntervals)
		{
			x_0 = inx_0;
			x_1 = inx_1;
			xIntervals = inxIntervals;
			dx = (x_1 - x_0) / xIntervals;
		
			// Simpson's rule requires an even number of intervals (= odd number of sample points)
			ALWAYS_ASSERT(!(xIntervals & 1));
		}
		
		virtual ~Integral2D() { }
		virtual ResultType Evaluate(const InnerEval *eval)
		{
			return this->InternalEvaluate(eval, xIntervals);
		}
	};

	template<class ResultType, class InnerEval, class KernelX, class KernelY, class KernelZ, bool moreAccuracy> class Integral3D : public Integral2D<ResultType, InnerEval, KernelY, KernelZ, moreAccuracy>
	{
		/*	This class performs a 3D integral (potentially using multiple threads to divide the work). 
			It should not be necessary to use this class directly - the functions at the end of this
			file such as Integrate, IntegrateOverCircle should generally be used	*/
	  protected:
		double x_0, x_1, dx;
		int xIntervals;
		typedef CallbackParms<ResultType, InnerEval, moreAccuracy> CBParmsType;
	  
		virtual void IntegerWorkCallback(int thisItem, void *userData)
		{
			// This version will be called in the case of a triple integral
			CBParmsType *cbParms = (CBParmsType *)userData;
			
			double x = x_0 + thisItem * dx;
			ResultType temp = Integral2D<ResultType, InnerEval, KernelY, KernelZ, moreAccuracy>::EvaluateMiddleIntegralUnthreaded(cbParms->eval, x) * KernelX::Evaluate(x) * SimpsonsMultiplier(thisItem, xIntervals) * dx;
			cbParms->AddResult(temp);
		}

	  public:
		Integral3D(double inx_0, double inx_1, int inxIntervals, double y_0, double y_1, int yIntervals, double z_0, double z_1, int zIntervals) : Integral2D<ResultType, InnerEval, KernelY, KernelZ, moreAccuracy>(y_0, y_1, yIntervals, z_0, z_1, zIntervals)
		{
			x_0 = inx_0;
			x_1 = inx_1;
			xIntervals = inxIntervals;
			dx = (x_1 - x_0) / xIntervals;
		
			// Simpson's rule requires an even number of intervals (= odd number of sample points)
			ALWAYS_ASSERT(!(xIntervals & 1));
		}
		
		virtual ~Integral3D() { }
		virtual ResultType Evaluate(const InnerEval *eval) { return this->InternalEvaluate(eval, xIntervals); }
	};
}

// Macros used to define functors required for use with the integration functions (below)
#define DEFINE_CLASS_FUNCTOR(TYPE, CLASS, METHOD, NAME) \
	class NAME \
	{ \
		const CLASS	*const theClass; \
	  public: \
		NAME(const CLASS * const in) : theClass(in) { } \
	    TYPE	Evaluate (double dummy, double dummy2, double x) const { return theClass->METHOD(x); } \
	};

#define DEFINE_CLASS_FUNCTOR_EXTRA_PARAM(TYPE, CLASS, METHOD, EXTRA_PARAM_TYPE, NAME) \
	class NAME \
	{ \
		const CLASS	*const theClass; \
		EXTRA_PARAM_TYPE extra; \
	  public: \
		NAME(const CLASS * const in, EXTRA_PARAM_TYPE inExtra) : theClass(in) { extra = inExtra; } \
	    TYPE	Evaluate (double dummy, double dummy2, double x) const { return theClass->METHOD(x, extra); } \
	};
	
#define DEFINE_CLASS_FUNCTOR_2_EXTRA_PARAMS(TYPE, CLASS, METHOD, EXTRA_PARAM_TYPE1, EXTRA_PARAM_TYPE2, NAME) \
    class NAME \
    { \
        const CLASS	*const theClass; \
        EXTRA_PARAM_TYPE1 extra1; \
        EXTRA_PARAM_TYPE2 extra2; \
      public: \
        NAME(const CLASS * const in, EXTRA_PARAM_TYPE1 inExtra1, EXTRA_PARAM_TYPE2 inExtra2) : theClass(in) { extra1 = inExtra1; extra2 = inExtra2; } \
        TYPE	Evaluate (double dummy, double dummy2, double x) const { return theClass->METHOD(x, extra1, extra2); } \
};

#define DEFINE_CLASS_2D_FUNCTOR(TYPE, CLASS, METHOD, NAME) \
	class NAME \
	{ \
		const CLASS	*const theClass; \
	  public: \
		NAME(const CLASS * const in) : theClass(in) { } \
	    TYPE	Evaluate (double dummy, double x, double y) const { return theClass->METHOD(x, y); } \
	};
	
#define DEFINE_CLASS_2D_FUNCTOR_EXTRA_PARAM(TYPE, CLASS, METHOD, EXTRA_PARAM_TYPE, NAME) \
	class NAME \
	{ \
		const CLASS	*const theClass; \
		EXTRA_PARAM_TYPE extra; \
	  public: \
		NAME(const CLASS * const in, EXTRA_PARAM_TYPE inExtra) : theClass(in) { extra = inExtra; } \
	    TYPE	Evaluate (double dummy, double x, double y) const { return theClass->METHOD(x, y, extra); } \
	};
	
#define DEFINE_CLASS_3D_FUNCTOR(TYPE, CLASS, METHOD, NAME) \
	class NAME \
	{ \
		const CLASS	*const theClass; \
	  public: \
		NAME(const CLASS * const in) : theClass(in) { } \
	    TYPE	Evaluate (double x, double y, double z) const { return theClass->METHOD(x, y, z); } \
	};

#define DEFINE_CLASS_3D_FUNCTOR_EXTRA_PARAM(TYPE, CLASS, METHOD, EXTRA_PARAM_TYPE, NAME) \
	class NAME \
	{ \
		const CLASS	*const theClass; \
		EXTRA_PARAM_TYPE extra; \
	  public: \
		NAME(const CLASS * const in, EXTRA_PARAM_TYPE inExtra) : theClass(in) { extra = inExtra; } \
	    TYPE	Evaluate (double x, double y, double z) const { return theClass->METHOD(x, y, z, extra); } \
	};
	
// Kernels used during integration
class UnitKernel
{
  public:
	static double	Evaluate(double x) { return 1.0; }
};

class CircularIntegralOuterKernel
{
  public:
	static double	Evaluate(double r) { return r; }
};

class SurfaceIntegralOuterKernel
{
  public:
	static double	Evaluate(double theta) { return sin(theta); }
};

class VolumeIntegralOuterKernel
{
  public:
	static double	Evaluate(double r) { return SQUARE(r); }
};

//	These are the actual functions which should normally be called to perform an integration
template<class RETURN_TYPE, class KERNEL_FUNC, class FUNC_TYPE> RETURN_TYPE Integrate(const FUNC_TYPE *functor, double x_0, double x_1, int xIntervals)
{
	jIntegralPrivate::Integral<RETURN_TYPE, FUNC_TYPE, KERNEL_FUNC, false> integral(x_0, x_1, xIntervals);
	return integral.Evaluate(functor);
}

template<class RETURN_TYPE, class KERNEL_FUNC, class FUNC_TYPE> RETURN_TYPE BetterSlowerIntegrate(const FUNC_TYPE *functor, double x_0, double x_1, int xIntervals)
{
	jIntegralPrivate::Integral<RETURN_TYPE, FUNC_TYPE, KERNEL_FUNC, true> integral(x_0, x_1, xIntervals);
	return integral.Evaluate(functor);
}

template<class RETURN_TYPE, class OUTER_KERNEL, class INNER_KERNEL, class FUNC_TYPE> RETURN_TYPE Integrate2D(const FUNC_TYPE *functor, double x_0, double x_1, int xIntervals, double y_0, double y_1, int yIntervals)
{
	jIntegralPrivate::Integral2D<RETURN_TYPE, FUNC_TYPE, OUTER_KERNEL, INNER_KERNEL, false> integral(x_0, x_1, xIntervals, y_0, y_1, yIntervals);
	return integral.Evaluate(functor);
}

template<class RETURN_TYPE, class OUTER_KERNEL, class MIDDLE_KERNEL, class INNER_KERNEL, class FUNC_TYPE> RETURN_TYPE Integrate3D(const FUNC_TYPE *functor, double x_0, double x_1, int xIntervals, double y_0, double y_1, int yIntervals, double z_0, double z_1, int zIntervals)
{
	jIntegralPrivate::Integral3D<RETURN_TYPE, FUNC_TYPE, OUTER_KERNEL, MIDDLE_KERNEL, INNER_KERNEL, false> integral(x_0, x_1, xIntervals, y_0, y_1, yIntervals, z_0, z_1, zIntervals);
	return integral.Evaluate(functor);
}

template<class RETURN_TYPE, class FUNC_TYPE> RETURN_TYPE IntegrateOverCircle(const FUNC_TYPE *functor, double r_1, int xIntervals, int yIntervals)
{
	return Integrate2D<RETURN_TYPE, CircularIntegralOuterKernel, UnitKernel>(functor, 0.0, r_1, xIntervals, 0.0, 2.0 * PI, yIntervals);
}

template<class RETURN_TYPE, class FUNC_TYPE> RETURN_TYPE IntegrateOverSurface2(const FUNC_TYPE &functor, int xIntervals, int yIntervals)
{
	return Integrate2D<RETURN_TYPE, SurfaceIntegralOuterKernel, UnitKernel>(&functor, 0.0, PI, xIntervals, 0.0, 2.0 * PI, yIntervals);
}

template<class RETURN_TYPE, class FUNC_TYPE> RETURN_TYPE IntegrateOverHemisphericalSurface(const FUNC_TYPE *functor, int xIntervals, int yIntervals)
{
	return Integrate2D<RETURN_TYPE, SurfaceIntegralOuterKernel, UnitKernel>(functor, 0.0, PI / 2.0, xIntervals, 0.0, 2.0 * PI, yIntervals);
}

template<class RETURN_TYPE, class FUNC_TYPE> RETURN_TYPE IntegrateOverSphericalVolume2(const FUNC_TYPE &functor, double r_1, int xIntervals, int yIntervals, int zIntervals)
{
	return Integrate3D<RETURN_TYPE, VolumeIntegralOuterKernel, SurfaceIntegralOuterKernel, UnitKernel>(&functor, 0.0, r_1, xIntervals, 0.0, PI, yIntervals, 0.0, 2.0 * PI, zIntervals);
}



#ifdef __JCOORD_H__
// This is probably not an ideal way of doing it, but I want to define this but this header
// still accessible to code that does not include coord3
template<class RETURN_TYPE, class FUNC_TYPE> RETURN_TYPE IntegrateOverCubicVolume(const FUNC_TYPE *functor, coord3 origin, double r, int xIntervals, int yIntervals, int zIntervals)
{
	return Integrate3D<RETURN_TYPE, UnitKernel, UnitKernel, UnitKernel>(functor, origin.x-r, origin.x+r, xIntervals, origin.y-r, origin.y+r, yIntervals, origin.z-r, origin.z+r, zIntervals);
}
#endif

#endif
