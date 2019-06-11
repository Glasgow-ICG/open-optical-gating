/*	Template: ObjectPool.h

	Copyright 2010-2015 Jonathan Taylor. All rights reserved.

	Template classes for maintaining a pool of allocated objects that can be reserved by individual threads.
	This can be useful when an object takes time to create, allocate, or in some way post-process, but
	we don't know in advance exactly how many will be required simultaneously.
	
	Examples include per-thread partial results for a numerical integral,
	and workspaces for translation matrix generation.
	
	Once an object pool is created (which automatically creates a specified number of objects in the pool),
	objects can be reserved from the pool using GetAvailableObject() and released using FinishedWithObject()
*/

#ifndef __OBJECT_POOL_H__
#define __OBJECT_POOL_H__ 1

template<class OBJ> class ObjectPool
{
  protected:
	OBJ 	*object;
	int		numObjects, objectsInUseMask, objectsUsedMask;
	
  public:
	ObjectPool(int num) : numObjects(num), objectsInUseMask(0), objectsUsedMask(0) { object = new OBJ[numObjects]; }
	virtual ~ObjectPool() { delete[] object; }
	
	void ResetPool(void)
	{
		ALWAYS_ASSERT(objectsInUseMask == 0);
		objectsUsedMask = 0;
	}
	
	int size(void) const { return numObjects; }
	OBJ *GetAvailableObject(bool *firstTimeForObject = NULL)
	{
		int lastReadOfBitfield = objectsInUseMask;
		int entryToUse = -1;
		do
		{
			// Identify an entry in the bitfield which appears to be free
			for (int i = 0; i < numObjects; i++)
			{
				if (!(lastReadOfBitfield & (1 << i)))
				{
					entryToUse = i;
					break;
				}
			}
			/*  There should always be at least one entry free, because we expect the client to create
				a sufficiently large pool for the number of threads that are running.
				If the client has chosen not to make a pool that big (maybe because of memory constraints)
				then at the moment it is up to the client to make sure they never request multiple
				objects simultaneously, to avoid hitting this assertion. There is currently no support
				in this class for waiting until an object becomes available, for example	*/
			ALWAYS_ASSERT(entryToUse != -1);
			// Attempt to reserve this entry
			lastReadOfBitfield = __sync_fetch_and_or(&objectsInUseMask, (1<<entryToUse));
			// Somebody else may have reserved it since we last read the bitfield,
			// so we may go round the loop again
		} while (lastReadOfBitfield & (1<<entryToUse));
		
		/*	If this is the first time we've used this entry, we must zero the coeffs
			Note that access to this bit in tempFieldCoeffsUsed will never be contested,
			but we must use sync access because other threads may be accessing other bits
			in this bitfield	*/
//		printf("Pool %p using obj[%d], bitfield now %x\n", this, entryToUse, objectsInUseMask);
		lastReadOfBitfield = __sync_fetch_and_or(&objectsUsedMask, (1<<entryToUse));
		if (firstTimeForObject != NULL)
		{
			if (!(lastReadOfBitfield & (1<<entryToUse)))
				*firstTimeForObject = true;
			else
				*firstTimeForObject = false;
		}
		
		ALWAYS_ASSERT(entryToUse < numObjects);
		return &object[entryToUse];
	}
	
	void FinishedWithObject(OBJ *obj)
	{
		int entry = obj - &object[0];
		int oldVal = __sync_fetch_and_and(&objectsInUseMask, ~(1 << entry));
		ALWAYS_ASSERT(oldVal & (1<<entry));		// Confirm that it was reserved beforehand
//		printf("Pool %p finished with obj[%d], bitfield now %x\n", this, entry, objectsInUseMask);
	}
	
	bool EntryWasUsed(int i)
	{
		return (objectsUsedMask & (1 << i));
	}

	int NumObjects(void) const { return numObjects; }

	OBJ *GetIndObject(int i)
	{
		// Note that this is of course not threadsafe, and should only be used for post-processing on the main thread
		ALWAYS_ASSERT(i < numObjects);
		return &object[i];
	}
	
	void MergeIntoDestination(OBJ *dest)
	{
		// Add every temp object that was used onto the destination object
		ALWAYS_ASSERT(objectsInUseMask == 0);
		for (int i = 0; i < numObjects; i++)
			if (EntryWasUsed(i))
			{
	//			printf("Pool %p merge obj[%d]\n", this, i);
				*dest += object[i];
			}
	}
};

#endif
