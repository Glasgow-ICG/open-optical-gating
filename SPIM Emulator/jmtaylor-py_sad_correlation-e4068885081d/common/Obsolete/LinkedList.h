#ifndef __LINKEDLIST_H__
#define __LINKEDLIST_H__

#include "jCommon.h"

class LinkedListElement
{
  private:
  	void				*self;
	LinkedListElement	*next;
	LinkedListElement	*prev;

  protected:
	// Subclasses may want to access the actual list elements
	LinkedListElement	*PrevNode(void) const { return(prev); }
	LinkedListElement	*NextNode(void) const { return(next); }
  public:
						// Element constructor
						LinkedListElement(void *inSelf) { self = inSelf; next = this; prev = this; }
						// List head constructor
						LinkedListElement(void) { self = 0L; next = this; prev = this; }
		virtual			~LinkedListElement()
						{
							// Don't throw exceptions from destructor
							HARMLESS_ASSERT(next == this);
							HARMLESS_ASSERT(prev == this);
						}
				void	InsertAfter(LinkedListElement	*before);
				void	InsertBefore(LinkedListElement	*after);
				void	RemoveFromList(void);
				void	*PrevL(void) const { return(prev->self); }
				void	*NextL(void) const { return(next->self); }
	LinkedListElement	*IterateToFindListHead(void);
				bool	InList(void) const { ASSERT(self != NULL); return(next != this); }
				bool	EmptyList(void) const { ASSERT(self == NULL); return(next == this); }
				void	MakeNewHead(LinkedListElement *oldHead);
				void	*Self(void) const { return(self); }

				// List iteration functions				
				long	GetListLength(void) const;
				void	PopulateArrayWithListElementSelves(void **array) const;
				void	MakeListEmpty(void);
};

template <class T> class TLLElement : public LinkedListElement
{
  public:
				TLLElement(void *inSelf) : LinkedListElement(inSelf) { }
				TLLElement(void) {}
		T		*PrevL(void) const { return (T*)LinkedListElement::PrevL(); }
		T		*NextL(void) const { return (T*)LinkedListElement::NextL(); }
		T		*Self(void) const { return (T*)LinkedListElement::Self(); }
				void	InsertBefore(TLLElement<T>	*before) { LinkedListElement::InsertBefore(before); }
				void	InsertAfter(TLLElement<T>	*after) { LinkedListElement::InsertAfter(after); }
};

#define ITERATE_OVER_LIST2(HEAD, CLASS, VARIABLE, NEXT_FN)		\
		for ((VARIABLE) = (HEAD).NextL();						\
				(VARIABLE) != NULL;								\
				(VARIABLE) = (VARIABLE)->NEXT_FN())

#define ITERATE_BACK_OVER_LIST2(HEAD, CLASS, VARIABLE, PREV_FN)		\
		for ((VARIABLE) = (HEAD).PrevL();						\
				(VARIABLE) != NULL;								\
				(VARIABLE) = (VARIABLE)->PREV_FN())

#define ITERATE_OVER_LIST(HEAD, CLASS, VARIABLE)				\
		ITERATE_OVER_LIST2((HEAD), (CLASS), (VARIABLE), NextL)

#define ITERATE_BACK_OVER_LIST(HEAD, CLASS, VARIABLE)				\
		ITERATE_BACK_OVER_LIST2((HEAD), (CLASS), (VARIABLE), PrevL)

#endif
