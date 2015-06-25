//
// Copyright (c) 2010-2011 Matthew Jack and Doug Binks
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

////////////////////////////////////////////////////////////////////////////////////////////////////////
// 
// IObject header file.
//
// The RuntimeCompiler library does not declare an IObject interface, only forward declares it.
// Hence each project can define their own base interface for objects they want to runtime compile
// and construct by using their own declaration of IObject in their own header file.
//
////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#ifndef IOBJECT_INCLUDED
#define IOBJECT_INCLUDED

#include "ObjectInterface.h"
#include "ISimpleSerializer.h"
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <mutex>

struct ISimpleSerializer;
class ObjectFactorySystem;
template<typename T> class shared_ptr;

// IIDs
enum InterfaceIDEnum
{
	IID_IOBJECT,
    IID_NodeObject,
    IID_Plotter,
	IID_ENDInterfaceID
};

typedef unsigned int InterfaceID;

// Template to help with IIDs
template< InterfaceID Tiid, typename TSuper> struct TInterface : public TSuper
{
	static const InterfaceID s_interfaceID = Tiid;
    virtual IObject* GetInterface( InterfaceID _iid)
    {
        switch(_iid)
        {
        case Tiid:
            return this;
            break;
        default:
            return TSuper::GetInterface(_iid);
        }
    }
};
class IObjectNotifiable
{
protected:
    friend struct IObject;
    virtual void updateObject(IObject* ptr) = 0;
};
template<typename T> class weak_ptr : public IObjectNotifiable
{
    T* m_object;
    friend struct IObject;
    friend class shared_ptr<T>;
    virtual void updateObject(IObject *ptr)
    {
        //m_object = dynamic_cast<T*>(ptr);
		m_object = (T*)ptr;
    }


public:
    weak_ptr(): m_object(nullptr)
    {
    }
    weak_ptr(IObject* ptr):
        m_object(dynamic_cast<T*>(ptr))
    {
        m_object->registerNotifier(this);
    }
    weak_ptr(T* ptr):
        m_object(ptr)
    {
        m_object->registerNotifier(this);
    }
    weak_ptr(weak_ptr const & ptr):
        m_object(nullptr)
    {
        swap(ptr);
    }
    ~weak_ptr()
    {
        if(m_object)
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
        if(m_object)
            m_object->deregisterNotifier(this);
        m_object = r.m_object;
        if(m_object)
            m_object->registerNotifier(this);
    }
    T* get() const
    {
        assert(m_object != nullptr);
        return m_object;
    }

};

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
        if(refCount)
        {
            (*refCount)--;
            if(*refCount <= 0)
            {
                delete refCount;
                delete m_object;
            }
        }
    }

    void increment()
    {
        if(refCount)
            ++(*refCount);
    }


public:
    shared_ptr(): m_object(nullptr), refCount(nullptr)
    {
    }
    shared_ptr(IObject* ptr):
        m_object(dynamic_cast<T*>(ptr)),
        refCount(new int)
    {
        *refCount = 1;
        m_object->registerNotifier(this);
    }
    shared_ptr(T* ptr):
        m_object(ptr),
        refCount(new int)
    {
        *refCount = 1;
        m_object->registerNotifier(this);
    }
    shared_ptr(shared_ptr const & ptr):
        shared_ptr()
    {
        swap(ptr);
    }
    template<typename V> shared_ptr(shared_ptr<V> const& ptr):
        shared_ptr()
    {
        swap(ptr);
    }


    ~shared_ptr()
    {
        if(m_object)
            m_object->deregisterNotifier(this);
        decrement();
    }
    T* operator->()
    {
        assert(m_object != nullptr);
        return m_object;
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
        if(m_object)
            m_object->deregisterNotifier(this);
         m_object = r.m_object;
        refCount = r.refCount;
        increment();
        if(m_object)
            m_object->registerNotifier(this);
    }
    template<typename V> void swap(shared_ptr<V> const & r)
    {
        decrement();
        if(m_object)
            m_object->deregisterNotifier(this);
        m_object = dynamic_cast<T*>(r.m_object);
        if(m_object == nullptr)
            return;
        refCount = r.refCount;
        increment();
        if(m_object)
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

// IObject itself below is a special case as the base class
// Also it doesn't hurt to have it coded up explicitly for reference

struct IObject
{
    static const InterfaceID s_interfaceID = IID_IOBJECT;

    virtual IObject* GetInterface(InterfaceID __iid)
    {
        switch(__iid)
        {
        case IID_IOBJECT:
            return this;
        default:
            return nullptr;
        }
    }

    template< typename T> void GetInterface( T** pReturn )
    {
        GetInterface( T::s_interfaceID, (void**)pReturn );
    }


    IObject() : _isRuntimeDelete(false) {}
    virtual ~IObject()
    {
        for(size_t i = 0; i < notifiers.size(); ++i)
        {
            notifiers[i]->updateObject(nullptr);
        }
    }

    // Perform any object initialization
    // Should be called with isFirstInit=true on object creation
    // Will automatically be called with isFirstInit=false whenever a system serialization is performed
    virtual void Init( bool isFirstInit )
    {

    }

    //return the PerTypeObjectId of this object, which is unique per class
    virtual PerTypeObjectId GetPerTypeId() const = 0;

    virtual void GetObjectId( ObjectId& id ) const
    {
        id.m_ConstructorId = GetConstructor()->GetConstructorId();
        id.m_PerTypeId = GetPerTypeId();
    }
    virtual ObjectId GetObjectId() const
    {
        ObjectId ret;
        GetObjectId( ret );
        return ret;
    }


    //return the constructor for this class
    virtual IObjectConstructor* GetConstructor() const = 0;

    //serialise is not pure virtual as many objects do not need state
    virtual void Serialize(ISimpleSerializer *pSerializer)
    {
        

    }
	void SerializeNotifiers(ISimpleSerializer* pSerializer)
	{
		SERIALIZE(notifiers);
	}
    virtual const char* GetTypeName() const = 0;

    virtual void registerNotifier(IObjectNotifiable* notifier)
    {
		std::lock_guard<std::recursive_mutex> lock(notifierMutex);
        auto itr = std::find(notifiers.begin(), notifiers.end(), notifier);
        if(itr == notifiers.end())
            notifiers.push_back(notifier);
        else
            return;
    }

    void deregisterNotifier(IObjectNotifiable* notifier)
    {
		std::lock_guard<std::recursive_mutex> lock(notifierMutex);
        auto itr = std::find(notifiers.begin(), notifiers.end(), notifier);
        if(itr != notifiers.end())
            notifiers.erase(itr);
        else
            return;
    }
    void updateNotifiers()
    {
		std::lock_guard<std::recursive_mutex> lock(notifierMutex);
        for(size_t i = 0; i < notifiers.size(); ++i)
        {
            if(notifiers[i] != nullptr)
                notifiers[i]->updateObject(this);
        }
    }
    virtual void updateParent(){}

    std::vector<IObjectNotifiable*> notifiers;
protected:
    bool IsRuntimeDelete() { return _isRuntimeDelete; }

private:
    friend class ObjectFactorySystem;

    // Set to true when object is being deleted because a new version has been created
    // Destructor should use this information to not delete other IObjects in this case
    // since these objects will still be needed
    bool _isRuntimeDelete;
	std::recursive_mutex notifierMutex;
};

#endif //IOBJECT_INCLUDED
