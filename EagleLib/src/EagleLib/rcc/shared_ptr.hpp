#pragma once
#include "IObjectNotifiable.h"
#include "IObject.h"
#include <cassert>
#include <EagleLib/Defs.hpp>

/**
*  The RCC_shared_ptr class is similar to boost::shared_ptr except that it auto updates the ptr when
*  an object swap is performed.  It does this by registering itself as updatable to the IObject
*/
namespace rcc
{
    EAGLE_EXPORTS IObject* get_object(ObjectId id);
    template<typename T> class weak_ptr;
    template<typename T> class shared_ptr: public IObjectNotifiable
    {
        T* m_object;
        int* refCount;
        ObjectId m_objectId;
        friend struct IObject;
        template<typename U> friend class weak_ptr;
        template<typename U> friend class rcc::shared_ptr;
    
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
        typedef T element_type;
        shared_ptr() :IObjectNotifiable(), m_object(nullptr), refCount(nullptr)
        {
        }
        shared_ptr(IObject* ptr) :
            IObjectNotifiable(),
            m_object(dynamic_cast<T*>(ptr)),
            refCount(new int)
        {
            *refCount = 1;
            m_objectId = m_object->GetObjectId();
        }
        shared_ptr(T* ptr) :
            IObjectNotifiable(),
            m_object(ptr),
            refCount(new int)
        {
            *refCount = 1;
            m_objectId = m_object->GetObjectId();
        }
        shared_ptr(shared_ptr const& ptr) :
            IObjectNotifiable(),
            m_object(ptr.m_object), refCount(ptr.refCount), m_objectId(ptr.m_objectId)
        {
            increment();
        }

        template<typename V> shared_ptr(shared_ptr<V> & ptr) :
            IObjectNotifiable(),
            m_object(dynamic_cast<T*>(ptr.m_object)), refCount(ptr.refCount), m_objectId(ptr.m_objectId)
        {
            increment();
        }

        ~shared_ptr()
        {
            decrement();
        }

        virtual void notify_swap()
        {
            m_object = nullptr;
            auto obj = get_object(m_objectId);
            if(obj)
                m_object = dynamic_cast<T*>(obj);
        }
        virtual void notify_delete()
        {
            // This should never happen
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
            decrement();
            m_object = r.m_object;
            m_objectId = r.m_objectId;
            refCount = r.refCount;
            increment();
            return *this;
        }
        template<typename V> shared_ptr& operator=(shared_ptr<V> const & r)
        {
            decrement();
            m_object = r.m_object;
            m_objectId = r.m_objectId;
            refCount = r.refCount;
            increment();
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
        bool operator == (shared_ptr const & r)
        {
            return r.m_object == m_object;
        }
        bool operator != (shared_ptr const& r)
        {
            return r.get() != m_object;
        }
        void reset(T* r = nullptr)
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
        void swap(shared_ptr & r)
        {
            decrement();
            auto obj = m_object; m_object = r.m_object; r.m_object = obj;
            auto count = refCount; refCount = r.refCount; r.refCount = count;
            auto id = m_objectId; m_objectId = r.m_objectId;
            increment();
        }
        template<typename V> void swap(shared_ptr<V> & r)
        {
            decrement();
            m_object = dynamic_cast<T*>(r.m_object);
            assert(m_object != nullptr);
            if (m_object == nullptr)
                return;
            auto obj = m_object; m_object = r.m_object; r.m_object = obj;
            auto count = refCount; refCount = r.refCount; r.refCount = count;
            auto id = m_objectId; m_objectId = r.m_objectId; 
            increment();
        }
        T* get() const
        {
            assert(m_object != nullptr);
            return m_object;
        }
        explicit operator bool() const
        {
            return m_object != nullptr;
        }
        ObjectId get_id() const
        {
            return m_objectId;
        }
    };

    template<typename T> class weak_ptr: public IObjectNotifiable
    {
        T* m_object;
        int* refCount;
        ObjectId m_objectId;
        friend struct IObject;
        friend class shared_ptr<T>;
    public:
        typedef T element_type;
        template<class U> weak_ptr(const shared_ptr<U>& other):
            IObjectNotifiable(),
            m_object(dynamic_cast<T*>(other.m_object)), refCount(other.refCount)
        {
        
        }
        weak_ptr() :IObjectNotifiable(), m_object(nullptr), refCount(nullptr)
        {
        }
        weak_ptr(IObject* ptr) :
            IObjectNotifiable(),
            m_object(dynamic_cast<T*>(ptr)), refCount(nullptr)
        {
            m_objectId = m_object->GetObjectId();
        }
        weak_ptr(T* ptr) :
            IObjectNotifiable(),
            m_object(ptr), refCount(nullptr)
        {
            m_objectId = m_object->GetObjectId();
        }
        weak_ptr(weak_ptr const & ptr) :
            IObjectNotifiable(),
            m_object(nullptr), refCount(nullptr)
        {
            swap(ptr);
        }
        ~weak_ptr()
        {
        }
        virtual void notify_swap()
        {
            m_object = nullptr;
            auto obj = get_object(m_objectId);
            if(obj)
                m_object = dynamic_cast<T*>(obj);
        }
        virtual void notify_delete()
        {
            m_object = nullptr;
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
        weak_ptr& operator=(IObject * r)
        {
            if(r == nullptr)
            {
                m_object = nullptr;
                m_objectId.SetInvalid();
                return *this;
            }
            m_object = dynamic_cast<T*>(r);
            m_objectId = m_object->GetObjectId();
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
            m_object = r.m_object;
            if(m_object)
                m_objectId = m_object->GetObjectId();
        }
        void reset(T* r = nullptr)
        {
            m_object = r;
            if(r)
            {
                m_objectId = r->GetObjectId();
            }
            else
            {
                m_objectId.SetInvalid();
            }
            refCount = nullptr;
        }
        T* get()
        {
            assert(m_object != nullptr);
            return m_object;
        }
        explicit operator bool() const
        {
            return m_object != nullptr;
        }
    };




    template<> class shared_ptr<IObject>: public IObjectNotifiable
    {
        IObject* m_object;
        ObjectId m_objectId;
        int* refCount;
        friend struct IObject;
        template<typename U> friend class weak_ptr;
        template<typename U> friend class rcc::shared_ptr;
        virtual void updateObject(IObject *ptr);
        void decrement();
        void increment();

    public:
        typedef IObject element_type;
        shared_ptr();
        shared_ptr(IObject* ptr);
        shared_ptr(shared_ptr<IObject> const & ptr);
        template<typename V> shared_ptr(shared_ptr<V> const& ptr) :
            shared_ptr()
        {
            swap(ptr);
        }
        ~shared_ptr();
        virtual void notify_swap();
        virtual void notify_delete();
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
        template<typename V> rcc::shared_ptr<IObject>& operator=(shared_ptr<V> const& r)
        {
            swap(r);
            return *this;
        }
        void reset(IObject* r);
        bool operator ==(IObject* p);
        bool operator !=(IObject* p);
        bool operator == (shared_ptr<IObject> const & r);
        bool operator != (shared_ptr<IObject> const& r);
        void swap(shared_ptr<IObject> const & r);
        template<typename V> void swap(shared_ptr<V> const & r)
        {
            decrement();
            m_object = dynamic_cast<IObject*>(r.m_object);
            assert(m_object != nullptr);
            if (m_object == nullptr)
                return;
            m_objectId = r.m_objectId;
            refCount = r.refCount;
            increment();
        }
        IObject* get();
        template<typename V> V* get()
        {
            assert(m_object != nullptr);
            return dynamic_cast<V*>(m_object);
        }
        explicit operator bool() const
        {
            return m_object != nullptr;
        }
    };

    template<> class weak_ptr<IObject> : public IObjectNotifiable
    {
        IObject* m_object;
        ObjectId m_objectId;
        friend struct IObject;
        template<typename T> friend class rcc::shared_ptr;
        virtual void updateObject(IObject *ptr);
    public:
        typedef IObject element_type;
        weak_ptr();
        weak_ptr(IObject* ptr);
        weak_ptr(weak_ptr<IObject> const & ptr);
        ~weak_ptr();
        virtual void notify_swap();
        virtual void notify_delete();
        IObject* operator->();
        weak_ptr<IObject>& operator=(weak_ptr const & r);
        bool operator ==(IObject* p);
        bool operator !=(IObject* p);
        bool operator == (weak_ptr<IObject> const & r);
        bool operator == (shared_ptr<IObject> const & r);
        bool operator != (weak_ptr const& r);
        bool operator !=(shared_ptr<IObject> const& r);
        void swap(weak_ptr<IObject> const & r);
        IObject* get();
        explicit operator bool() const
        {
            return m_object != nullptr;
        }
    };
}
