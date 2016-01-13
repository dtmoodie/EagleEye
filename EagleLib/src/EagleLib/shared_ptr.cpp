#include "shared_ptr.hpp"
#include "IObject.h"
void shared_ptr<IObject>::updateObject(IObject *ptr)
{
    m_object = static_cast<IObject*>(ptr);
}
void shared_ptr<IObject>::decrement()
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

void shared_ptr<IObject>::increment()
{
    if (refCount)
        ++(*refCount);
}


 shared_ptr<IObject>::shared_ptr() : m_object(nullptr), refCount(nullptr)
{
}
 shared_ptr<IObject>::shared_ptr(IObject* ptr) :
    m_object(ptr),
    refCount(new int)
{
    *refCount = 1;
    m_object->registerNotifier(this);
}

 shared_ptr<IObject>::shared_ptr(shared_ptr<IObject> const & ptr) :
    shared_ptr()
{
    swap(ptr);
}


 shared_ptr<IObject>::~shared_ptr()
{
    if (m_object)
        m_object->deregisterNotifier(this);
    decrement();
}

 IObject* shared_ptr<IObject>::operator->()
{
    assert(m_object != nullptr);
    return m_object;
}

 shared_ptr<IObject>& shared_ptr<IObject>::operator=(shared_ptr<IObject> const & r)
{
    swap(r);
    return *this;
}

 bool shared_ptr<IObject>::operator ==(IObject* p)
{
    return m_object == p;
}
 bool shared_ptr<IObject>::operator !=(IObject* p)
{
    return m_object != p;
}
 bool shared_ptr<IObject>::operator == (shared_ptr<IObject> const & r)
{
    return r.get() == m_object;
}
 bool shared_ptr<IObject>::operator != (shared_ptr<IObject> const& r)
{
    return r.get() != m_object;
}
 void shared_ptr<IObject>::swap(shared_ptr<IObject> const & r)
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
   
 IObject* shared_ptr<IObject>::get() const
{
    assert(m_object != nullptr);
    return m_object;
}


 void weak_ptr<IObject>::updateObject(IObject *ptr)
 {
     m_object = ptr;
 }


 weak_ptr<IObject>::weak_ptr() : m_object(nullptr)
    {
    }
 weak_ptr<IObject>::weak_ptr(IObject* ptr) :
        m_object(ptr)
    {
        m_object->registerNotifier(this);
    }
 weak_ptr<IObject>::weak_ptr(weak_ptr<IObject> const & ptr) :
        m_object(nullptr)
    {
        swap(ptr);
    }
 weak_ptr<IObject>::~weak_ptr()
    {
        if (m_object)
            m_object->deregisterNotifier(this);
    }
    IObject* weak_ptr<IObject>::operator->()
    {
        assert(m_object != nullptr);
        return m_object;
    }
    weak_ptr<IObject>& weak_ptr<IObject>::operator=(weak_ptr const & r)
    {
        swap(r);
        return *this;
    }
    bool weak_ptr<IObject>::operator ==(IObject* p)
    {
        return m_object == p;
    }
    bool weak_ptr<IObject>::operator !=(IObject* p)
    {
        return m_object != p;
    }
    bool weak_ptr<IObject>::operator == (weak_ptr<IObject> const & r)
    {
        return r.get() == m_object;
    }
    bool weak_ptr<IObject>::operator == (shared_ptr<IObject> const & r)
    {
        return r.get() == m_object;
    }

    bool weak_ptr<IObject>::operator != (weak_ptr const& r)
    {
        return r.get() != m_object;
    }
    bool weak_ptr<IObject>::operator !=(shared_ptr<IObject> const& r)
    {
        return r.get() != m_object;
    }

    void weak_ptr<IObject>::swap(weak_ptr<IObject> const & r)
    {
        if (m_object)
            m_object->deregisterNotifier(this);
        m_object = r.m_object;
        if (m_object)
            m_object->registerNotifier(this);
    }
    IObject* weak_ptr<IObject>::get() const
    {
        assert(m_object != nullptr);
        return m_object;
    }