#pragma once

#include <list>
#include <mutex>
#include <atomic>
#include <cassert>

namespace EagleLib
{
    namespace pool
    {
        template<typename T> class Ptr;
        // Basic object pool class, objects are removed and added from the pool as they are released by the Ptr class
        template<typename T> class ObjectPool
        {
        public:
            ObjectPool()
            {
            }
            ~ObjectPool()
            {
                for (auto itr = _pool.begin(); itr != _pool.end(); ++itr)
                {
                    delete *itr;
                }
                _pool.clear();
            }
            virtual Ptr<T> get_object()
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (_pool.empty())
                {
                    _pool.push_back(new T);
                }
                auto itr = _pool.begin();
                auto ret =  Ptr<T>(*itr, this);
                _pool.erase(itr);
                return ret;
            }
        protected:
            std::list<T*> _pool;
            std::mutex mtx;
            template<typename U> friend class Ptr;
        };

        template<typename T> class Ptr
        {
            int* _refCount;
            T* _ptr;
            ObjectPool<T>* _pool;
        public:
            Ptr()
            {
                _refCount = nullptr;
                _ptr = nullptr;
                _pool = nullptr;
            }
            Ptr(T* ptr, ObjectPool<T>* pool)
            {
                _refCount = new int;
                *_refCount = 1;
                _ptr = ptr;
                _pool = pool;
            }
            
            Ptr(const Ptr<T>& other)
            {
                _refCount = nullptr;
                _ptr = nullptr;
                _pool = nullptr;
                swap(other);
            }

            ~Ptr()
            {
                decrement();
            }
            Ptr& operator=(Ptr const & r)
            {
                swap(r);
                return *this;
            }
            T* operator->()
            {
                assert(_ptr != nullptr);
                return _ptr;
            }
            T* get() const
            {
                assert(_ptr != nullptr);
                return _ptr;
            }
            operator T*() const {return get();}
            
            void decrement()
            {
                if (_refCount)
                {
                    (*_refCount)--;
                    if (*_refCount <= 0)
                    {
                        std::lock_guard<std::mutex> lock(_pool->mtx);
                        _pool->_pool.push_back(_ptr);
                        delete _refCount;
                    }
                }
            }
            void increment()
            {
                if (_refCount)
                    ++(*_refCount);
            }
            void swap(const Ptr<T>& other)
            {
                decrement();
                _ptr = other._ptr;
                _pool = other._pool;
                _refCount = other._refCount;
                increment();
            }
        };
    }
}
