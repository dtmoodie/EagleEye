#pragma once

#include "EagleLib/Defs.hpp"
#include "IObject.h"

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include "type.h"

#include <memory>
#include <list>
#include <vector>
#include <mutex>

class ISimpleSerializer;
namespace Parameters
{
	class Parameter;
	template<typename T> class ITypedParameter;
	template<typename T> class TypedInputParameter;
}
namespace Signals
{
	class connection;
	template<typename T> class default_combiner;
	template<class T, template<class> class C> class typed_signal_base;
	
}
namespace cv
{
	class FileNode;
	namespace cuda
	{
		class Stream;
	}
}
namespace EagleLib
{
	class IVariableManager;
	class DataStream;
    struct ParameteredObjectImpl; // Private implementation stuffs
	class SignalManager;
	typedef std::shared_ptr<Parameters::Parameter> ParameterPtr;

    class EAGLE_EXPORTS ParameteredObject
    {
    public:
        ParameteredObject();
        ~ParameteredObject();
		virtual void setup_signals(SignalManager* manager);
		virtual void SetupVariableManager(IVariableManager* manager);
        virtual IVariableManager* GetVariableManager();
        virtual void onUpdate(cv::cuda::Stream* stream = nullptr);
		virtual Parameters::Parameter* addParameter(ParameterPtr param);
        virtual void RemoveParameter(std::string name);
        virtual void RemoveParameter(size_t index);
        virtual bool exists(const std::string& name);
        virtual bool exists(size_t index);
        // Thows exception on unable to get parameter
		virtual ParameterPtr getParameter(int idx);
		virtual ParameterPtr getParameter(const std::string& name);

        // Returns nullptr on unable to get parameter
		virtual ParameterPtr getParameterOptional(int idx);
		virtual ParameterPtr getParameterOptional(const std::string& name);
        virtual std::vector<ParameterPtr> getParameters();


        virtual void RegisterParameterCallback(int idx, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param = false, bool lock_object = false);
        virtual void RegisterParameterCallback(const std::string& name, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param = false, bool lock_object = false);
        virtual void RegisterParameterCallback(Parameters::Parameter* param, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param = false, bool lock_object = false);

        template<typename T>
        Parameters::Parameter* registerParameter(const std::string& name, T* data);

        template<typename T>
        Parameters::Parameter* addParameter(const std::string& name, const T& data);
        
        template<typename T>
        Parameters::Parameter* addIfNotExist(const std::string& name, const T& data);

        template<typename T>
        Parameters::TypedInputParameter<T>* addInputParameter(const std::string& name);

        template<typename T>
        bool updateInputQualifier(const std::string& name, const std::function<bool(Parameters::Parameter*)>& qualifier);

        template<typename T>
        bool updateInputQualifier(int idx, const std::function<bool(Parameters::Parameter*)>& qualifier);

        template<typename T>
        Parameters::Parameter* updateParameterPtr(const std::string& name, T* data, cv::cuda::Stream* stream = nullptr);

        template<typename T>
        Parameters::Parameter* updateParameter(const std::string& name, const T& data, cv::cuda::Stream* stream = nullptr);

        template<typename T>
        Parameters::Parameter* updateParameter(size_t idx, const T data, cv::cuda::Stream* stream = nullptr);

        template<typename T>
        typename std::shared_ptr<Parameters::ITypedParameter<T>> getParameter(std::string name);

        template<typename T>
		typename std::shared_ptr<Parameters::ITypedParameter<T>> getParameter(int idx);

        template<typename T>
		typename std::shared_ptr<Parameters::ITypedParameter<T>> getParameterOptional(std::string name);

        template<typename T>
		typename std::shared_ptr<Parameters::ITypedParameter<T>> getParameterOptional(int idx);

        
        // Mutex for blocking processing of a node during parameter update
        std::recursive_mutex                                              mtx;
    protected:
		IVariableManager*				                                                    _variable_manager;
		Signals::typed_signal_base<void(ParameteredObject*), Signals::default_combiner>*    _sig_parameter_updated;
		Signals::typed_signal_base<void(ParameteredObject*), Signals::default_combiner>*    _sig_parameter_added;
        std::list<std::shared_ptr<Signals::connection>>                                     _callback_connections;
        std::vector<ParameterPtr>                                                           _parameters;
    private:
        void RunCallbackLockObject(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback);
        void RunCallbackLockParameter(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback, std::recursive_mutex* paramMtx);
        void RunCallbackLockBoth(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback, std::recursive_mutex* paramMtx);
    };

    class EAGLE_EXPORTS ParameteredIObject: public IObject, public ParameteredObject
    {
    public:
        ParameteredIObject();
        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual void Init(const cv::FileNode& configNode);
		virtual void Init(bool firstInit);
        
    };
}
