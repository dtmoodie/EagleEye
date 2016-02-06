#pragma once
#include "IObjectNotifiable.h"
#include "IObject.h"
#include <cassert>
/**
*  The RCC_shared_ptr class is similar to boost::shared_ptr except that it auto updates the ptr when
*  an object swap is performed.  It does this by registering itself as updatable to the IObject
*/
template<typename T> class shared_ptr : public IObjectNotifiable
{
	T* m_object;
	int* refCount;
	friend struct IObject;
	template<typename U> friend class weak_ptr;
	template<typename U> friend class shared_ptr;
	virtual void updateObject(IObject *ptr)
	{
		m_object = static_cast<T*>(ptr);

	}
	void decrement()
	{
		if (refCount)
		{
			(*refCount)--;
			if (*refCount <= 0)
			{
				delete refCount;
				delete m_object;
			}
		}
	}

	void increment()
	{
		if (refCount)
			++(*refCount);
	}
    

public:
	shared_ptr() : m_object(nullptr), refCount(nullptr)
	{
	}
	shared_ptr(IObject* ptr) :
		m_object(dynamic_cast<T*>(ptr)),
		refCount(new int)
	{
		*refCount = 1;
		m_object->registerNotifier(this);
	}
	shared_ptr(T* ptr) :
		m_object(ptr),
		refCount(new int)
	{
		*refCount = 1;
		m_object->registerNotifier(this);
	}
	shared_ptr(shared_ptr const & ptr) :
		shared_ptr()
	{
		swap(ptr);
	}

	template<typename V> shared_ptr(shared_ptr<V> const& ptr) :
		shared_ptr()
	{
		swap(ptr);
	}

	~shared_ptr()
	{
		if (m_object)
			m_object->deregisterNotifier(this);
		decrement();
	}

	T* operator->()
	{
		assert(m_object != nullptr);
		return m_object;
	}

    template<typename U> U* staticCast()
    {
        return static_cast<U*>(m_object);
    }

    template<typename U> U* dynamicCast()
    {
        return dynamic_cast<U*>(m_object);
    }

	shared_ptr& operator=(shared_ptr const & r)
	{
		swap(r);
		return *this;
	}
	template<typename V> shared_ptr& operator=(shared_ptr<V> const& r)
	{
		swap(r);
		return &this;
	}

	bool operator ==(T* p)
	{
		return m_object == p;
	}
	bool operator !=(T* p)
	{
		return m_object != p;
	}
	bool operator == (shared_ptr const & r)
	{
		return r.get() == m_object;
	}
	bool operator != (shared_ptr const& r)
	{
		return r.get() != m_object;
	}
	void swap(shared_ptr const & r)
	{
		decrement();
		if (m_object)
			m_object->deregisterNotifier(this);
		m_object = r.m_object;
		refCount = r.refCount;
		increment();
		if (m_object)
			m_object->registerNotifier(this);
	}
	template<typename V> void swap(shared_ptr<V> const & r)
	{
		decrement();
		if (m_object)
			m_object->deregisterNotifier(this);
		m_object = dynamic_cast<T*>(r.m_object);
        assert(m_object != nullptr);
		if (m_object == nullptr)
			return;
		refCount = r.refCount;
		increment();
		if (m_object)
			m_object->registerNotifier(this);
	}
	T* get() const
	{
		assert(m_object != nullptr);
		return m_object;
	}
	template<typename V> V* get()
	{
		assert(m_object != nullptr);
		return dynamic_cast<V*>(m_object);
	}
};

template<typename T> class weak_ptr : public IObjectNotifiable
{
	T* m_object;
	friend struct IObject;
	friend class shared_ptr<T>;
	virtual void updateObject(IObject *ptr)
	{
		m_object = static_cast<T*>(ptr);
	}


public:
	weak_ptr() : m_object(nullptr)
	{
	}
	weak_ptr(IObject* ptr) :
		m_object(dynamic_cast<T*>(ptr))
	{
		m_object->registerNotifier(this);
	}
	weak_ptr(T* ptr) :
		m_object(ptr)
	{
		m_object->registerNotifier(this);
	}
	weak_ptr(weak_ptr const & ptr) :
		m_object(nullptr)
	{
		swap(ptr);
	}
	~weak_ptr()
	{
		if (m_object)
			m_object->deregisterNotifier(this);
	}
	T* operator->()
	{
		assert(m_object != nullptr);
		return m_object;
	}
	weak_ptr& operator=(weak_ptr const & r)
	{
		swap(r);
		return *this;
	}
	bool operator ==(T* p)
	{
		return m_object == p;
	}
	bool operator !=(T* p)
	{
		return m_object != p;
	}
	bool operator == (weak_ptr const & r)
	{
		return r.get() == m_object;
	}
	bool operator == (shared_ptr<T> const & r)
	{
		return r.get() == m_object;
	}

	bool operator != (weak_ptr const& r)
	{
		return r.get() != m_object;
	}
	bool operator !=(shared_ptr<T> const& r)
	{
		return r.get() != m_object;
	}

	void swap(weak_ptr const & r)
	{
		if (m_object)
			m_object->deregisterNotifier(this);
		m_object = r.m_object;
		if (m_object)
			m_object->registerNotifier(this);
	}
	T* get() const
	{
		assert(m_object != nullptr);
		return m_object;
	}

};




template<> class shared_ptr<IObject> : public IObjectNotifiable
{
    IObject* m_object;
    int* refCount;
    friend struct IObject;
    template<typename U> friend class weak_ptr;
    template<typename U> friend class shared_ptr;
    virtual void updateObject(IObject *ptr);
    void decrement();

    void increment();


public:
    shared_ptr();
    shared_ptr(IObject* ptr);
    shared_ptr(shared_ptr<IObject> const & ptr);

    template<typename V> shared_ptr(shared_ptr<V> const& ptr) :
        shared_ptr()
    {
        swap(ptr);
    }

    ~shared_ptr();

    IObject* operator->();

    template<typename U> U* staticCast()
    {
        return static_cast<U*>(m_object);
    }

    template<typename U> U* dynamicCast()
    {
        return dynamic_cast<U*>(m_object);
    }

    shared_ptr& operator=(shared_ptr<IObject> const & r);
    template<typename V> shared_ptr<IObject>& operator=(shared_ptr<V> const& r)
    {
        swap(r);
        return *this;
    }

    bool operator ==(IObject* p);
    bool operator !=(IObject* p);
    bool operator == (shared_ptr<IObject> const & r);
    bool operator != (shared_ptr<IObject> const& r);
    void swap(shared_ptr<IObject> const & r);
    template<typename V> void swap(shared_ptr<V> const & r)
    {
        decrement();
        if (m_object)
            m_object->deregisterNotifier(this);
        m_object = dynamic_cast<IObject*>(r.m_object);
        assert(m_object != nullptr);
        if (m_object == nullptr)
            return;
        refCount = r.refCount;
        increment();
        if (m_object)
            m_object->registerNotifier(this);
    }
    IObject* get() const;
    template<typename V> V* get()
    {
        assert(m_object != nullptr);
        return dynamic_cast<V*>(m_object);
    }
};

template<> class weak_ptr<IObject> : public IObjectNotifiable
{
    IObject* m_object;
    friend struct IObject;
    template<typename T> friend class shared_ptr;
    virtual void updateObject(IObject *ptr);
public:
    weak_ptr();
    weak_ptr(IObject* ptr);
    weak_ptr(weak_ptr<IObject> const & ptr);
    ~weak_ptr();
    IObject* operator->();
    weak_ptr<IObject>& operator=(weak_ptr const & r);
    bool operator ==(IObject* p);
    bool operator !=(IObject* p);
    bool operator == (weak_ptr<IObject> const & r);
    bool operator == (shared_ptr<IObject> const & r);

    bool operator != (weak_ptr const& r);
    bool operator !=(shared_ptr<IObject> const& r);

    void swap(weak_ptr<IObject> const & r);
    IObject* get() const;
};
