#pragma once
#include <boost/shared_ptr.hpp>

namespace Parameters
{
    class Parameter
    {

    };

    template<typename T, typename TypePolicy, typename ItemPolicy = void> class ParameterBase: public Parameter, public TypePolicy, ItemPolicy
    {

    };

    template<typename T> class ITypedParameter: public Parameter
    {
    public:
        virtual T& Data() = 0;
        virtual void UpdateData(const T& data_) = 0;
    };
    template<typename T> class TypedParameterPolicy: public Parameter
    {
        T data;
    public:
        virtual T& Data()
        {
            return data;
        }
        virtual void UpdateData(const T& data_)
        {
            data = data_;
        }
    };

    template<typename T> class PointerParameterPolicy: public ITypedParameter<T>
    {
        T* ptr;
    public:
        PointerParameterPolicy(): ptr(nullptr){}
        virtual T& Data()
        {
            if(ptr)
                return *ptr;
            else
                throw std::string("Nullptr");
        }
        virtual void UpdateData(T& data)
        {
            ptr = &data;
        }
    };





}
