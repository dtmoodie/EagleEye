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
	public:
		enum ParameterTypes
		{
			Input = 1,
			Output = 2,
			State = 4,
			Control = 8
		};
		Parameter(const std::string& name, const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = "");
		typedef boost::shared_ptr<Parameter> Ptr;
		virtual Loki::TypeInfo GetTypeInfo() = 0;

		virtual std::string& GetName();
		virtual void SetName(const std::string& name_);
		virtual const std::string& GetTooltip();
		virtual void SetTooltip(const std::string& tooltip_);
		virtual const std::string& GetTreeName();
		virtual void SetTreeName(const std::string& treeName_);
		virtual boost::signals2::connection RegisterNotifier(const boost::function<void(void)>& f);

	protected:
		boost::signals2::signal<void(void)> UpdateSignal;
	private:
		std::string name;
		std::string tooltip;
		std::string treeName;
		ParameterTypes type;
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
		ITypedParameter(const std::string& name, const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = "") :
			Parameter(name, type, tooltip){}
		virtual Loki::TypeInfo GetTypeInfo()
		{
			return Loki::TypeInfo(typeid(T));
		}
		/*bool operator ==(const ITypedParameter& lhs, const Parameter& rhs)
		{
			if (lhs.GetType() == rhs.GetType())
			{
				auto typedRhs = dynamic_cast<ITypedParameter<T>>(rhs);
				return *Data() == *typedRhs.Data();
			}
			return false;
		}*/
    };

	template<typename T, template<typename> class Policy1 = Persistence::PersistencePolicy, template<typename> class Policy2 = UI::UiPolicy> 
	class MetaTypedParameter : 
		public ITypedParameter<T>, public Policy1<T>, public Policy2<T>
	{
	public:
		MetaTypedParameter(const std::string& name, const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = ""):
			ITypedParameter<T>(name, type, tooltip){}
	};


    template<typename T> class TypedParameter: public MetaTypedParameter<T>
    {
        T data;
    public:
		typedef boost::shared_ptr<TypedParameter<T>> Ptr;
		static Ptr create(const T& init, const std::string& name, const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = "")
		{
			return Ptr(new TypedParameter<T>(name, init, type, tooltip));
		}
		TypedParameter(const std::string& name, const T& init = T(), const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = ""):
			MetaTypedParameter<T>(name, type, tooltip), data(init) {}

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
		TypedParameterRef(const std::string& name, const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = ""):
			MetaTypedParameter<T>(name, type, tooltip){}
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
		static Ptr create(const std::string& name, T* ptr_ = nullptr,
			const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = "")
		{
			return Ptr(new TypedInputParameterPtr(name, ptr_, type, tooltip));
		}
		TypedParameterPtr(const std::string& name, T* ptr_ = nullptr, 
			const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = "") : 
			ptr(ptr_),
			MetaTypedParameter<T>(name, type, tooltip){}

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
		TypedInputParameter(const std::string& name, const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = ""):
			MetaTypedParameter<T>(name, type, tooltip){}
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
		TypedInputParameterPtr(const std::string& name, T** userVar_, 
			const ParameterTypes& type = ParameterTypes::Control, const std::string& tooltip = ""):
			MetaTypedParameter<T>(name, type, tooltip), userVar(userVar_){}

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
