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
    struct ParameteredObjectImpl; // Private implementation stuffs
	class SignalManager;
	typedef std::shared_ptr<Parameters::Parameter> ParameterPtr;
    class EAGLE_EXPORTS ParameteredObject
    {
    public:
        ParameteredObject();
        ~ParameteredObject();
		virtual void setup_signals(SignalManager* manager);
        virtual void onUpdate(cv::cuda::Stream* stream = nullptr);
		virtual Parameters::Parameter* addParameter(ParameterPtr param);
        bool exists(const std::string& name);
        bool exists(size_t index);
        // Thows exception on unable to get parameter
		ParameterPtr getParameter(int idx);
		ParameterPtr getParameter(const std::string& name);

        // Returns nullptr on unable to get parameter
		ParameterPtr getParameterOptional(int idx);
		ParameterPtr getParameterOptional(const std::string& name);


        void RegisterParameterCallback(int idx, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param = false, bool lock_object = false);
        void RegisterParameterCallback(const std::string& name, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param = false, bool lock_object = false);
        void RegisterParameterCallback(Parameters::Parameter* param, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param = false, bool lock_object = false);

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

        std::vector<ParameterPtr> parameters;
        // Mutex for blocking processing of a node during parameter update
        std::recursive_mutex                                              mtx;
		
		
    protected:
		std::list<std::shared_ptr<Signals::connection>> _callback_connections;
		Signals::typed_signal_base<void(ParameteredObject*), Signals::default_combiner>* _sig_parameter_updated;
		Signals::typed_signal_base<void(ParameteredObject*), Signals::default_combiner>* _sig_parameter_added;
    private:
		
        void RunCallbackLockObject(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback);
        void RunCallbackLockParameter(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback, std::recursive_mutex* paramMtx);
        void RunCallbackLockBoth(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback, std::recursive_mutex* paramMtx);
    };

    class EAGLE_EXPORTS ParameteredIObject: public IObject, public ParameteredObject
    {
    public:
        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual void Init(const cv::FileNode& configNode);
		virtual void Init(bool firstInit);
        
    };


}
