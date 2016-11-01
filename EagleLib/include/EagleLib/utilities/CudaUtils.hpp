/*
Copyright(c) 2015, Daniel Moodie
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met :

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and / or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CUDA_UTILS__HPP
#define CUDA_UTILS__HPP

#include <opencv2/core/cuda.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <queue>
#include <boost/thread.hpp>
#ifndef __CUDA_ARCH__
#include <MetaObject/Logging/Log.hpp>
#endif
namespace EagleLib
{

template<typename T> void cleanup(T& ptr, typename std::enable_if< std::is_array<T>::value>::type* = 0) { /*delete[] ptr;*/ }
template<typename T> void cleanup(T& ptr, typename std::enable_if< std::is_pointer<T>::value && !std::is_array<T>::value>::type* = 0) { delete ptr; }
template<typename T> void cleanup(T& ptr, typename std::enable_if<!std::is_pointer<T>::value && !std::is_array<T>::value>::type* = 0) { return; }

template<typename Data>
    class concurrent_notifier
    {
    private:
        boost::condition_variable the_condition_variable;
        std::vector<Data> the_data;
        mutable boost::mutex the_mutex;
    public:
        void wait_for_data()
        {
            boost::mutex::scoped_lock lock(the_mutex);
            while(the_data.empty())
            {
                the_condition_variable.wait(lock);
            }
        }
        void wait_push(Data const& data)
        {
            boost::mutex::scoped_lock lock(the_mutex);
            while (!the_data.empty()) // Wait till the consumer pulls data from the queue
            {
                the_condition_variable.wait(lock);
            }
            the_data.push_back(data);
        }

        void push(Data const& data)
        {
            boost::mutex::scoped_lock lock(the_mutex);
            bool const was_empty=the_data.empty();
            if(the_data.size())
                the_data[0] = data;
            else
                the_data.push_back(data);

            lock.unlock(); // unlock the mutex

            if(was_empty)
            {
                the_condition_variable.notify_one();
            }
        }
        void wait_and_pop(Data& popped_value)
        {
            boost::mutex::scoped_lock lock(the_mutex);
            while(the_data.empty())
            {
                the_condition_variable.wait(lock);
            }

            popped_value=the_data[0];
            the_data.clear();
        }
        bool try_pop(Data& popped_value)
        {
            boost::mutex::scoped_lock lock(the_mutex);
            if(the_data.empty())
                return false;
            popped_value= the_data[0];
            the_data.clear();
            return true;
        }

        size_t size()
        {
            boost::mutex::scoped_lock lock(the_mutex);
            return the_data.size();
        }
        void clear()
        {
            boost::mutex::scoped_lock lock(the_mutex);
            the_data.clear();
        }
    };

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
        void wait_push(Data const& data)
        {
            boost::mutex::scoped_lock lock(the_mutex);
            while (!the_queue.empty()) // Wait till the consumer pulls data from the queue
            {
                the_condition_variable.wait(lock);
            }
            the_queue.push_back(data);
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
        void clear()
        {
            boost::mutex::scoped_lock lock(the_mutex);
            the_queue = std::queue<Data>();
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
            return true;
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
    
    template<typename...> struct Buffer;

    template<typename T>
    struct Buffer<T>
    {
        T data;
        Buffer(){}
        Buffer(const T& init): data(init){}
        ~Buffer()
        {            cleanup(data);        }
        bool ready()
        {            return true;        }
        bool wait()
        {            return true;        }
        bool record(cv::cuda::Stream& stream)
        {            return true;        }
    };
    template<typename T, typename P1>
    struct Buffer<T,P1> : public P1
    {
        T data;
        Buffer() {}
        Buffer(const T& init) : data(init) {}
        ~Buffer()
        {            cleanup(data);        }
        bool ready()
        {            return P1::ready();        }
        bool wait()
        {            return P1::wait();        }
        bool record(cv::cuda::Stream& stream)
        {            return P1::record(stream);        }
    };
    template<typename T, typename P1, typename P2>
    struct Buffer<T,P1,P2> : public P1, public P2
    {
        T data;
        Buffer() {}
        Buffer(const T& init) : data(init) {}
        ~Buffer()
        {            cleanup(data);        }
        bool ready()
        {            return P1::ready() && P2::ready();        }
        bool wait()
        {            return P1::wait() && P2::wait();        }
        bool record(cv::cuda::Stream& stream)
        {            return P1::record(stream) || P2::record(stream);        }
    };

    template<typename...> class BufferPool;

    template<typename T> class BufferPool<T>
    {
        size_t getItr;
        size_t putItr;
        size_t size;
        std::vector<Buffer<T>> buffer;
        boost::recursive_mutex mtx;
    public:
        BufferPool<T>():getItr(0), putItr(0), size(10), buffer(10){}
        BufferPool<T>(size_t size_) :
            buffer(size_), size(size_), getItr(0), putItr(0){}
        BufferPool<T>(size_t size_, const T& init) :
            buffer(size_, Buffer<T>(init)), size(size_), getItr(0), putItr(0){}
        BufferPool<T>(const BufferPool<T>& other) :
            buffer(other.buffer), getItr(other.getItr), putItr(other.putItr), size(other.size){}
        Buffer<T>* getFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if(size == 0)
                return nullptr;
            auto itr = putItr++%size;
            if (!buffer[itr].ready())
            {
#ifndef __NVCC__
                LOG(warning) << "Buffer not ready, increasing size of buffer pool";
#endif
                resize(buffer.size() + 1);
            }
            return &buffer[itr];
        }
        Buffer<T>* getBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            auto itr = getItr++%size;
            if (!buffer[itr].ready())
            {
#ifndef __NVCC__
                LOG(warning) << "Buffer not ready, increasing size of buffer pool";
#endif
                resize(buffer.size() + 1);
            }
            return &buffer[itr];
        }
        Buffer<T>* waitBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            buffer[getItr%size].wait();
            return &buffer[getItr++%size];
        }
        Buffer<T>* waitFront()
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
        BufferPool& operator =(const BufferPool<T>& rhs)
        {
            getItr = rhs.getItr;
            putItr = rhs.putItr;
            size = rhs.size;
            buffer = rhs.buffer;
            return *this;
        }
    };
    template<typename T, typename P1> class BufferPool<T,P1>
    {
        size_t getItr;
        size_t putItr;
        size_t size;
        std::vector<Buffer<T, P1>> buffer;
        boost::recursive_mutex mtx;
    public:
        BufferPool() :getItr(0), putItr(0), size(10), buffer(10) {}
        BufferPool(size_t size_) :
            buffer(size_), size(size_), getItr(0), putItr(0) {}
        BufferPool(size_t size_, const T& init) :
            buffer(size_, Buffer<T, P1>(init)), size(size_), getItr(0), putItr(0) {}
        BufferPool(const BufferPool<T, P1>& other) :
            buffer(other.buffer), getItr(other.getItr), putItr(other.putItr), size(other.size) {}
        Buffer<T, P1>* getFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if (size == 0)
                return nullptr;
            auto itr = putItr++%size;
            if (!buffer[itr].ready())
            {
#ifndef __NVCC__
                LOG(warning) << "Buffer not ready, increasing size of buffer pool";
#endif
                resize(buffer.size() + 1);
            }
            return &buffer[itr];
        }
        Buffer<T, P1>* getBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            auto itr = getItr++%size;
            if (!buffer[itr].ready())
            {
#ifndef __NVCC__
                LOG(warning) << "Buffer not ready, increasing size of buffer pool";
#endif
                resize(buffer.size() + 1);
            }            
            return &buffer[itr];
        }
        Buffer<T, P1>* waitBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            buffer[getItr%size].wait();
            return &buffer[getItr++%size];
        }
        Buffer<T, P1>* waitFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            buffer[putItr%size].wait();
            return &buffer[putItr++%size];
        }
        void resize(size_t size_)
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            size = size_;
            buffer.resize(size);
        }
        BufferPool& operator =(const BufferPool<T, P1>& rhs)
        {
            getItr = rhs.getItr;
            putItr = rhs.putItr;
            size = rhs.size;
            buffer = rhs.buffer;
            return *this;
        }
    };
    template<typename T, typename P1, typename P2> class BufferPool<T,P1,P2>
    {
        size_t getItr;
        size_t putItr;
        size_t size;
        std::vector<Buffer<T, P1, P2>> buffer;
        boost::recursive_mutex mtx;
    public:
        BufferPool() :getItr(0), putItr(0), size(10), buffer(10) {}
        BufferPool(size_t size_) :
            buffer(size_), size(size_), getItr(0), putItr(0) {}
        BufferPool(size_t size_, const T& init) :
            buffer(size_, Buffer<T, P1, P2>(init)), size(size_), getItr(0), putItr(0) {}
        BufferPool(const BufferPool<T, P1, P2>& other) :
            buffer(other.buffer), getItr(other.getItr), putItr(other.putItr), size(other.size) {}
        Buffer<T, P1, P2>* getFront()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            if (size == 0)
                return nullptr;
            auto itr = putItr++%size;
            if (!buffer[itr].ready())
            {
#ifndef __NVCC__
                LOG(warning) << "Buffer not ready, increasing size of buffer pool";
#endif
                resize(buffer.size() + 1);
            }
            return &buffer[itr];
        }
        Buffer<T, P1, P2>* getBack()
        {
            boost::recursive_mutex::scoped_lock lock(mtx);
            auto itr = getItr++%size;
            if (!buffer[itr].ready())
            {
#ifndef __NVCC__
                LOG(warning) << "Buffer not ready, increasing size of buffer pool";
#endif
                resize(buffer.size() + 1);
            }
            return &buffer[itr];
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
            size = size_;
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
            buffer(size_), size(size_), getItr(0), putItr(0){}
        ConstEventBuffer(size_t size_, const T& init):
            buffer(size_, EventBuffer<T>(init)), size(size_), getItr(0), putItr(0){}
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
            buffer(size_), size(size_), getItr(0), putItr(0){}
        ConstHostBuffer(size_t size_, const T& init):
            buffer(size_, LockedBuffer<T>(init)), size(size_), getItr(0), putItr(0){}
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
            return *this;
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
            buffer(size_), size(size_), getItr(0), putItr(0){}
        ConstBuffer(size_t size_, const T& init):
            buffer(size_, init), size(size_), getItr(0), putItr(0){}
        ConstBuffer(const ConstEventBuffer<T>& other):
            buffer(other.buffer), getItr(other.getItr), putItr(other.putItr), size(other.size){}
        ~ConstBuffer()
        {
            for(size_t i = 0; i < buffer.size(); ++i)
            {
                EagleLib::cleanup(buffer[i]);
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

#endif
