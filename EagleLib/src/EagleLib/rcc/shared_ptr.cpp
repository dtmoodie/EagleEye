#include "shared_ptr.hpp"
#include "IObject.h"
#include <EagleLib/rcc/ObjectManager.h>

using namespace rcc;

IObject* rcc::get_object(ObjectId id)
{
    return EagleLib::ObjectManager::Instance().GetObject(id);
}

void rcc::shared_ptr<IObject>::notify_swap()
{
    m_object = nullptr;
    auto obj = get_object(m_objectId);
    if(obj)
        m_object = obj;
}
void rcc::shared_ptr<IObject>::notify_delete()
{

}
void rcc::shared_ptr<IObject>::updateObject(IObject *ptr)
{
    m_object = static_cast<IObject*>(ptr);
}
void rcc::shared_ptr<IObject>::decrement()
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
void rcc::shared_ptr<IObject>::reset(IObject* r)
{
    decrement();
    m_object = r;
    if(r)
    {
        refCount = new int;
        m_objectId = r->GetObjectId();
    }
    else
    {
        refCount = nullptr;
        m_objectId.SetInvalid();
    }
    increment();
}

void rcc::shared_ptr<IObject>::increment()
{
    if (refCount)
        ++(*refCount);
}


 rcc::shared_ptr<IObject>::shared_ptr(): IObjectNotifiable(), m_object(nullptr), refCount(nullptr)
{
}
 rcc::shared_ptr<IObject>::shared_ptr(IObject* ptr) : IObjectNotifiable(),
    m_object(ptr),
    refCount(new int)
{
    *refCount = 1;
}

 rcc::shared_ptr<IObject>::shared_ptr(shared_ptr<IObject> const & ptr) :
    shared_ptr()
{
    swap(ptr);
}


 rcc::shared_ptr<IObject>::~shared_ptr()
{
    decrement();
}

 IObject* rcc::shared_ptr<IObject>::operator->()
{
    assert(m_object != nullptr);
    return m_object;
}

 rcc::shared_ptr<IObject>& rcc::shared_ptr<IObject>::operator=(shared_ptr<IObject> const & r)
{
    swap(r);
    return *this;
}

 bool rcc::shared_ptr<IObject>::operator ==(IObject* p)
{
    return m_object == p;
}
 bool rcc::shared_ptr<IObject>::operator !=(IObject* p)
{
    return m_object != p;
}
 bool rcc::shared_ptr<IObject>::operator == (shared_ptr<IObject> const & r)
{
    return r.m_objectId == m_objectId;
}
 bool rcc::shared_ptr<IObject>::operator != (shared_ptr<IObject> const& r)
{
    return r.m_objectId != m_objectId;
}
 void rcc::shared_ptr<IObject>::swap(shared_ptr<IObject> const & r)
{
    decrement();
    m_object = r.m_object;
    refCount = r.refCount;
    increment();
}
   
 IObject* rcc::shared_ptr<IObject>::get()
{
    assert(m_object != nullptr);
    return m_object;
}


 void weak_ptr<IObject>::updateObject(IObject *ptr)
 {
     m_object = ptr;
 }


weak_ptr<IObject>::weak_ptr() :IObjectNotifiable(), m_object(nullptr)
{
}
weak_ptr<IObject>::weak_ptr(IObject* ptr) :IObjectNotifiable(),
    m_object(ptr)
{
    
}
weak_ptr<IObject>::weak_ptr(weak_ptr<IObject> const & ptr) :IObjectNotifiable(),
    m_object(nullptr)
{
    swap(ptr);
}
weak_ptr<IObject>::~weak_ptr()
{
    
}
void weak_ptr<IObject>::notify_swap()
{
    m_object = nullptr;
    auto obj = get_object(m_objectId);
    if(obj)
        m_object = obj;
}
void rcc::weak_ptr<IObject>::notify_delete()
{
	m_object = nullptr;
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
    return r.m_objectId == m_objectId;
}
bool weak_ptr<IObject>::operator == (shared_ptr<IObject> const & r)
{
    return r.m_objectId == m_objectId;
}

bool weak_ptr<IObject>::operator != (weak_ptr const& r)
{
    return r.m_objectId != m_objectId;
}
bool weak_ptr<IObject>::operator !=(shared_ptr<IObject> const& r)
{
    return r.m_objectId != m_objectId;
}

void weak_ptr<IObject>::swap(weak_ptr<IObject> const & r)
{
    m_object = r.m_object;   
}
IObject* weak_ptr<IObject>::get()
{
    assert(m_object != nullptr);
    return m_object;
}