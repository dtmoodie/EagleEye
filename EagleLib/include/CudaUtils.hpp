#pragma once

#include <opencv2/core/cuda.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <queue>
#include <boost/thread.hpp>
namespace EagleLib
{

template<typename T> void cleanup(T ptr, typename std::enable_if<std::is_pointer<T>::value>::type* = 0) { delete ptr; }
template<typename T> void cleanup(T ptr, typename std::enable_if<!std::is_pointer<T>::value>::type* = 0){ return; }
    template<typename Data>
    class concurrent_queue
    {
    private:
        boost::condition_variable the_condition_variable;
        std::queue<Data> the_queue;
        mutable boost::mutex the_mutex;
    public:
        void wait_for_data()
        {
            boost::mutex::scoped_lock lock(the_mutex);
            while(the_queue.empty())
            {
                the_condition_variable.wait(lock);
            }
        }
        void push(Data const& data)
        {
            boost::mutex::scoped_lock lock(the_mutex);
            bool const was_empty=the_queue.empty();
            the_queue.push(data);

            lock.unlock(); // unlock the mutex

            if(was_empty)
            {
                the_condition_variable.notify_one();
            }
        }
        void wait_and_pop(Data& popped_value)
        {
            boost::mutex::scoped_lock lock(the_mutex);
            while(the_queue.empty())
            {
                the_condition_variable.wait(lock);
            }

            popped_value=the_queue.front();
            the_queue.pop();
        }
        bool try_pop(Data& popped_value)
        {
            boost::mutex::scoped_lock lock(the_mutex);
            if(the_queue.empty())
                return false;
            popped_value=the_queue.front();
            the_queue.pop();
            return true;
        }

        size_t size()
        {
            boost::mutex::scoped_lock lock(the_mutex);
            return the_queue.size();
        }
    };

    struct EventPolicy
    {
        cv::cuda::Event fillEvent;
        bool ready()
        {
            return fillEvent.queryIfComplete();
        }
        bool wait()
        {
            fillEvent.waitForCompletion();
        }
        bool record(cv::cuda::Stream& stream)
        {
            fillEvent.record(stream);
            return true;
        }
    };
    struct LockedPolicy
    {
        boost::recursive_mutex mtx;
        bool ready()
        {
            bool res = mtx.try_lock();
            if(res)
                mtx.unlock();
            return res;
        }
        bool wait()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            return true;
        }
        bool record(cv::cuda::Stream& stream)
        {
            return false;
        }

    };
    struct NullPolicy
    {
        bool ready()
        {
            return true;
        }
        bool wait()
        {
            return true;
        }
        bool record(cv::cuda::Stream& stream)
        {
            return false;
        }
    };

    template<typename T, typename P1 = NullPolicy, typename P2 = NullPolicy>
    struct Buffer: public P1, public P2
    {
        T data;
        Buffer(){}
        Buffer(const T& init): data(init){}
        ~Buffer()
        {            cleanup(data);        }
        bool ready()
        {
            return P1::ready() && P2::ready();
        }
        bool wait()
        {
            return P1::wait() && P2::wait();
        }
        bool record(cv::cuda::Stream& stream)
        {
            return P1::record(stream) || P2::record(stream);
        }

    };

    template<typename T, typename P1 = NullPolicy, typename P2 = NullPolicy> class BufferPool
    {
        size_t getItr;
        size_t putItr;
        size_t size;
        std::vector<Buffer<T, P1, P2>> buffer;
        boost::recursive_mutex mtx;
    public:
        BufferPool():buffer(10), getItr(0), putItr(0), size(10){}
        BufferPool(size_t size_):
            buffer(size_), size(size_), getItr(0), putItr(0), size(size_){}
        BufferPool(size_t size_, const T& init):
            buffer(size_, Buffer<T, P1, P2>(init)), size(size_), getItr(0), putItr(0), size(size_){}
        BufferPool(const BufferPool<T, P1, P2>& other):
            buffer(other.buffer), getItr(other.getItr), putItr(other.putItr), size(other.size){}
        Buffer<T, P1, P2>* getFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if(size == 0)
                return nullptr;
            return &buffer[putItr++%size];
        }
        Buffer<T, P1, P2>* getBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if(!buffer[getItr%size].ready())
                return nullptr;
            return &buffer[getItr++%size];
        }
        Buffer<T, P1, P2>* waitBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            buffer[getItr%size].wait();
            return &buffer[getItr++%size];
        }
        Buffer<T, P1, P2>* waitFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            buffer[putItr%size].wait();
            return &buffer[putItr++%size];
        }
        void resize(size_t size_)
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            size= size_;
            buffer.resize(size);
        }
        BufferPool& operator =(const BufferPool<T, P1, P2>& rhs)
        {
            getItr = rhs.getItr;
            putItr = rhs.putItr;
            size = rhs.size;
            buffer = rhs.buffer;
            return *this;
        }
    };
















    template<typename T> struct EventBuffer
    {
        T data;
        cv::cuda::Event fillEvent;
        EventBuffer(){}
        EventBuffer(const T& init): data(init){}
        ~EventBuffer()
        {
            cleanup(data);
        }

        bool ready()
        {
            return fillEvent.queryIfComplete();
        }
        void wait()
        {
            fillEvent.waitForCompletion();
        }
    };

    template<typename T> class LockedBuffer
    {
    public:
        LockedBuffer(const T& init):
            data(init){}
        ~LockedBuffer()
        {
            cleanup(data);
        }


        T data;
        boost::recursive_mutex mtx;
    };





    template<typename T> class ConstEventBuffer
    {
        std::vector<EventBuffer<T>> buffer;
        size_t getItr;
        size_t putItr;
        size_t size;
        boost::recursive_mutex mtx;
    public:
        ConstEventBuffer():buffer(10), getItr(0), putItr(0), size(10){}
        ConstEventBuffer(size_t size_):
            buffer(size_), size(size_), getItr(0), putItr(0), size(size_){}
        ConstEventBuffer(size_t size_, const T& init):
            buffer(size_, EventBuffer<T>(init)), size(size_), getItr(0), putItr(0), size(size_){}
        ConstEventBuffer(const ConstEventBuffer<T>& other):
            buffer(other.buffer), getItr(other.getItr), putItr(other.putItr), size(other.size){}

        EventBuffer<T>* getFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if(size == 0)
                return nullptr;
            return &buffer[putItr++%size];
        }

        EventBuffer<T>* getBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if(!buffer[getItr%size].ready())
                return nullptr;
            return &buffer[getItr++%size];
        }
        EventBuffer<T>* waitBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            buffer[getItr%size].wait();
            return &buffer[getItr++%size];
        }
        void resize(size_t size_)
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            size= size_;
            buffer.resize(size);
        }
        ConstEventBuffer& operator =(const ConstEventBuffer<T>& rhs)
        {
            getItr = rhs.getItr;
            putItr = rhs.putItr;
            size = rhs.size;
            buffer = rhs.buffer;
			return *this;
        }
    };


    template<typename T> class ConstHostBuffer
    {
        std::vector<LockedBuffer<T>> buffer;
        size_t getItr;
        size_t putItr;
        size_t size;
        boost::recursive_mutex mtx;
    public:
        ConstHostBuffer():buffer(10), getItr(0), putItr(0), size(10){}
        ConstHostBuffer(size_t size_):
            buffer(size_), size(size_), getItr(0), putItr(0), size(size_){}
        ConstHostBuffer(size_t size_, const T& init):
            buffer(size_, LockedBuffer<T>(init)), size(size_), getItr(0), putItr(0), size(size_){}
        ConstHostBuffer(const ConstEventBuffer<T>& other):
            buffer(other.buffer), getItr(other.getItr), putItr(other.putItr), size(other.size){}
        ~ConstHostBuffer()
        {
        }

        LockedBuffer<T>* getFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if(size == 0)
                return nullptr;
            return &buffer[putItr++%size];
        }

        LockedBuffer<T>* getBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            return &buffer[getItr++%size];
        }
        void resize(size_t size_)
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            size= size_;
            buffer.resize(size);
        }
        void resize(size_t size_, const T& init)
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            size= size_;
            buffer.resize(size, init);
        }
        ConstHostBuffer& operator =(const ConstHostBuffer<T>& rhs)
        {
            getItr = rhs.getItr;
            putItr = rhs.putItr;
            size = rhs.size;
            buffer = rhs.buffer;
        }
    };

    template<typename T> class ConstBuffer
    {
        std::vector<T> buffer;
        size_t getItr;
        size_t putItr;
        size_t size;
        boost::recursive_mutex mtx;
    public:
        ConstBuffer():buffer(10), getItr(0), putItr(0), size(10){}
        ConstBuffer(size_t size_):
            buffer(size_), size(size_), getItr(0), putItr(0), size(size_){}
        ConstBuffer(size_t size_, const T& init):
            buffer(size_, init), size(size_), getItr(0), putItr(0), size(size_){}
        ConstBuffer(const ConstEventBuffer<T>& other):
            buffer(other.buffer), getItr(other.getItr), putItr(other.putItr), size(other.size){}
        ~ConstBuffer()
        {
            for(int i = 0; i < buffer.size(); ++i)
            {
                cleanup(buffer[i]);
            }
        }

        T* getFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if(size == 0)
                return nullptr;
            return &buffer[putItr++%size];
        }

        T* getBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            return &buffer[getItr++%size];
        }
        void resize(size_t size_)
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            size= size_;
            buffer.resize(size);
        }
        void resize(size_t size_, const T& init)
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            size= size_;
            buffer.resize(size, init);
        }

        ConstBuffer& operator =(const ConstBuffer<T>& rhs)
        {
            getItr = rhs.getItr;
            putItr = rhs.putItr;
            size = rhs.size;
            buffer = rhs.buffer;
        }
    };
}

