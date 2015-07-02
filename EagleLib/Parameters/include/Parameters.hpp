#pragma once
#include <boost/shared_ptr.hpp>
#include <boost/signals2.hpp>
#include "LokiTypeInfo.h"
#include "UI/UI.hpp"
#include "Persistence/Persistence.hpp"

namespace Parameters
{
    class Parameter
    {
		std::string name;
	protected:
		boost::signals2::signal<void(void)> UpdateSignal;
	public:
		enum ParameterTypes
		{
			Input = 1,
			Output = 2,
			State = 4,
			Control = 8
		};
		typedef boost::shared_ptr<Parameter> Ptr;
		virtual Loki::TypeInfo GetTypeInfo() = 0;
		virtual const std::string& GetName()
		{
			return name;
		}
		virtual void SetName(const std::string& name_)
		{
			name = name_;
		}
		virtual boost::signals2::connection RegisterNotifier(const boost::function<void(void)>& f)
		{
			return UpdateSignal.connect(f);
		}
    };
	class InputParameter
	{
	public:
		typedef boost::shared_ptr<InputParameter> Ptr;
		virtual bool SetInput(const std::string& name_) = 0;
		virtual bool SetInput(const Parameter::Ptr param) = 0;
		virtual bool AcceptsInput(const Parameter::Ptr param) = 0;
		virtual bool AcceptsType(const Loki::TypeInfo& type) = 0;
	};

    template<typename T> class ITypedParameter: 
			public Parameter
    {
    public:
		typedef boost::shared_ptr<ITypedParameter<T>> Ptr;
        virtual T* Data() = 0;
		virtual void UpdateData(T& data_) = 0;
		virtual void UpdateData(const T& data_) = 0;
		virtual void UpdateData(T* data_) = 0;

		virtual Loki::TypeInfo GetTypeInfo()
		{
			return Loki::TypeInfo(typeid(T));
		}
    };

	template<typename T, 
		template<typename> class Policy1 = PersistencePolicy,
		template<typename> class Policy2 = UiPolicy> class MetaTypedParameter : 
		public ITypedParameter<T>, public Policy1<T>, public Policy2<T>
	{
	public:
		MetaTypedParameter():
			PersistencePolicy<T>,
			UiPolicy<T>{}
	};


    template<typename T> class TypedParameter: public MetaTypedParameter<T>
    {
        T data;
    public:
		typedef boost::shared_ptr<TypedParameter<T>> Ptr;
		static Ptr create(const T& init)
		{
			return Ptr(new TypedParameter<T>(init));
		}
		TypedParameter() {}
		TypedParameter(const T& init) : data(init) {}
        virtual T* Data()
        {
            return &data;
        }
        virtual void UpdateData(T& data_)
        {
            data = data_;
			UpdateSignal();
        }
		virtual void UpdateData(const T& data_)
		{
			data = data_;
			UpdateSignal();
		}
		virtual void UpdateData(T* data_)
		{
			data = *data_;
			UpdateSignal();
		}
    };
	
	template<typename T> class TypedParameterRef : public MetaTypedParameter<T>
	{
		T& data;
	public:
		typedef boost::shared_ptr<TypedParameterRef<T>> Ptr;

		virtual T* Data()
		{
			return &data;
		}
		virtual void UpdateData(const T& data_)
		{
			data = data_;
			UpdateSignal();
		}
		virtual void UpdateData(const T* data_)
		{
			data = *data_;
			UpdateSignal();
		}

	};

    template<typename T> class TypedParameterPtr: public MetaTypedParameter<T>
    {
        T* ptr;
    public:
		typedef boost::shared_ptr<TypedParameterPtr<T>> Ptr;
		static Ptr create(T* ptr_)
		{
			return Ptr(new TypedInputParameterPtr(ptr_));
		}
		TypedParameterPtr(): ptr(nullptr){}
		TypedParameterPtr(T* ptr_) : ptr(ptr_) {}
        virtual T* Data()
        {
            return ptr;   
        }
        virtual void UpdateData(T& data)
        {
            ptr = &data;
			UpdateSignal();
        }
		virtual void UpdateData(const T& data)
		{
			if (ptr)
				*ptr = data;
		}
		virtual void UpdateData(T* data_)
		{
			ptr = data_;
			UpdateSignal();
		}
    };
	// Meant for internal use and access, ie access through TypedInputParameter::Data()
	template<typename T> class TypedInputParameter : public MetaTypedParameter<T>, public InputParameter
	{
		ITypedParameter<T>::Ptr input;
		boost::signals2::connection inputConnection;
		virtual void onInputUpdate()
		{
		}
	public:
		typedef boost::shared_ptr<TypedInputParameter<T>> Ptr;
		virtual bool SetInput(const std::string& name_)
		{
			return false;
		}

		virtual bool SetInput(const Parameter::Ptr param)
		{
			ITypedParameter<T>::Ptr castedParam = boost::dynamic_pointer_cast<ITypedParameter<T>, Parameter>(param);
			if (castedParam)
			{
				input = castedParam;
				inputConnection.disconnect();
				inputConnection = castedParam->RegisterNotifier(boost::bind(&TypedInputParameter<T>::onInputUpdate, this));
				return true;
			}
			return false;
		}

		virtual bool AcceptsInput(const Parameter::Ptr param)
		{
			return Loki::TypeInfo(typeid(T)) == param->GetTypeInfo();
		}

		virtual bool AcceptsType(const Loki::TypeInfo& type)
		{
			return Loki::TypeInfo(typeid(T)) == type;
		}
		virtual T* Data()
		{
			if (input)
				return input->Data();
			return nullptr;
		}
		virtual void UpdateData(T& data_)
		{

		}
		virtual void UpdateData(const T& data_)
		{

		}
		virtual void UpdateData(T* data_)
		{

		}
	};

	// Meant to reference a pointer variable in user space, and to update that variable whenever 
	// IE int* myVar; 
	// auto typedParam = TypedInputParameterPtr(&myVar); // TypedInputParameter now updates myvar to point to whatever the
	// input variable is for typedParam.
	template<typename T> class TypedInputParameterPtr : public MetaTypedParameter<T>, public InputParameter
	{
		T** userVar; // Pointer to the user space pointer variable of type T
		ITypedParameter<T>::Ptr input;
		boost::signals2::connection inputConnection;
		virtual void onInputUpdate()
		{
			// The input variable has been updated, update user var
			*userVar = input->Data();
		}
	public:
		typedef boost::shared_ptr<TypedInputParameterPtr<T>> Ptr;
		static Ptr create(T** userVar_)
		{
			return Ptr(new TypedInputParameterPtr(userVar_));
		}
		TypedInputParameterPtr(T** userVar_)
		{
			userVar = userVar_;
		}

		virtual bool SetInput(const std::string& name_)
		{
			return false;
		}

		virtual bool SetInput(const Parameter::Ptr param)
		{
			ITypedParameter<T>::Ptr castedParam = boost::dynamic_pointer_cast<ITypedParameter<T>, Parameter>(param);
			if (castedParam)
			{
				input = castedParam;
				inputConnection.disconnect();
				inputConnection = castedParam->RegisterNotifier(boost::bind(&TypedInputParameterPtr<T>::onInputUpdate, this));
				*userVar = input->Data();
				return true;
			}
			return false;
		}

		virtual bool AcceptsInput(const Parameter::Ptr param)
		{
			return Loki::TypeInfo(typeid(T)) == param->GetTypeInfo();
		}

		virtual bool AcceptsType(const Loki::TypeInfo& type)
		{
			return Loki::TypeInfo(typeid(T)) == type;
		}
		virtual T* Data()
		{
			if (input)
				return input->Data();
			return nullptr;
		}
		virtual void UpdateData(T& data_)
		{

		}
		virtual void UpdateData(const T& data_)
		{

		}
		virtual void UpdateData(T* data_)
		{

		}

	};




}
