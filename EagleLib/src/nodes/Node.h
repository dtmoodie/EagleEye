#pragma once

/*
 *  Nodes are the base processing object of this library.  Nodes can encapsulate all aspects of image processing from
 *  a single operation to a parallel set of intercommunicating processing stacks.  Nodes achieve this through one or more
 *  of the following concepts:
 *
 *  1) Processing node - Input image -> output image
 *  2) Function node - publishes a boost::function object to be used by sibling nodes
 *  3) Object node - publishes an object to be used by sibling nodes
 *  4) Serial collection nodes - Input image gets processed by all child nodes in series
 *  5) Parallel collection nodes - Input image gets processed by all child nodes in parallel. One thread per node
 *
 *  Nodes should be organized in a tree structure.  Each node will be accessible by name from the top of the tree via /parent/...../treeName where
 *  treeName is the unique name associated with that node.  The parameters of that node can be accessed via /parent/...../treeName:name.
 *  Nodes should be iterable by their parents by insertion order.  They should be accessible by sibling nodes.
 *
 *  Parameters:
 *  - Four main types of parameters, input, output, control and status.
 *  -- Input parameters should be defined in the expecting node by their internal name, datatype and the name of the output assigned to this input
 *  -- Output parameters should be defined by their datatype, memory location and
 *  - Designs:
 *  -- Could have a vector of parameter objects for each type.  The parameter object would contain the tree name of the parameter,
 *      the parameter type, and a pointer to the parameter
 *  -- Other considerations include using a boost::shared_ptr to the parameter, in which case constructing node and any other node that uses the parameter would share access.
 *      this has the advantage that the parameters don't need to be updated when an object swap occurs, since they aren't deleted.
 *      This would be nice for complex parameter objects, but it has the downside of functors not being updated correctly, which isn't such a big deal because the
 *      developer should just update functors accordingly in the init(bool) function.
 *
*/


#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/signals2.hpp>
#include <boost/thread.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/attributes/scoped_attribute.hpp>
#include <boost/log/expressions/keyword.hpp>

#include <vector>
#include <type_traits>
#include "type.h" // for demangle on linux
#include "LokiTypeInfo.h"

#include <Parameters.hpp>
#include <Types.hpp>

#include <opencv2/core/cuda.hpp>
#include <external_includes/cv_core.hpp>
#include <external_includes/cv_highgui.hpp>

#include "CudaUtils.hpp"

#include "RuntimeLinkLibrary.h"
#include "ObjectInterfacePerModule.h"
#include "IObject.h"

/*

*/
#define TIME if(profile) timings.push_back(std::pair<clock_t, int>(clock(), __LINE__));
#ifdef _MSC_VER
#define NODE_LOG(severity) BOOST_LOG_SCOPED_THREAD_ATTR("NodeName", boost::log::attributes::constant<std::string>(fullTreeName));			\
	BOOST_LOG_SCOPED_THREAD_ATTR("Node", boost::log::attributes::constant<const Node*>(this));													\
	LOG_TRIVIAL(severity)
#else
#define NODE_LOG(severity) 	LOG_TRIVIAL(severity)
#endif

namespace EagleLib
{
	class Node;
}

BOOST_LOG_ATTRIBUTE_KEYWORD(NodeName, "NodeName",const std::string);
BOOST_LOG_ATTRIBUTE_KEYWORD(NodePtr, "Node", const EagleLib::Node*);


#ifdef _MSC_VER
#ifdef _DEBUG
	RUNTIME_COMPILER_LINKLIBRARY("EagleLibd.lib")
	RUNTIME_COMPILER_LINKLIBRARY("libParameterd.lib")
	RUNTIME_COMPILER_LINKLIBRARY("Qt5Cored.lib");
	RUNTIME_COMPILER_LINKLIBRARY("Qt5Networkd.lib");
	RUNTIME_COMPILER_LINKLIBRARY("Qt5Guid.lib");
	RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgetsd.lib");
#else
	RUNTIME_COMPILER_LINKLIBRARY("EagleLib.lib")
	RUNTIME_COMPILER_LINKLIBRARY("libParameter.lib")
	RUNTIME_COMPILER_LINKLIBRARY("Qt5Core.lib");
	RUNTIME_COMPILER_LINKLIBRARY("Qt5Network.lib");
	RUNTIME_COMPILER_LINKLIBRARY("Qt5Gui.lib");
	RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgets.lib");
#endif

#else
#ifdef _DEBUG
    RUNTIME_COMPILER_LINKLIBRARY("-lEagleLibd")
#else
	RUNTIME_COMPILER_LINKLIBRARY("-lEagleLib")
#endif
#endif

#define CATCH_MACRO                                                         \
}catch (boost::thread_resource_error& err)                                  \
{                                                                           \
    NODE_LOG(error)<< err.what();                                           \
}catch (boost::thread_interrupted& err)                                     \
{                                                                           \
    NODE_LOG(error)<<"Thread interrupted";                                  \
    /* Needs to pass this back up to the chain to the processing thread.*/  \
    /* That way it knowns it needs to exit this thread */                   \
    throw err;                                                              \
}catch (boost::thread_exception& err)                                       \
{                                                                           \
    NODE_LOG(error)<< err.what();                                           \
}                                                                           \
catch (cv::Exception &err)                                                  \
{                                                                           \
    NODE_LOG(error)<< err.what();                                           \
}                                                                           \
catch (boost::exception &err)                                               \
{                                                                           \
    NODE_LOG(error)<<"Boost error";                                         \
}catch(std::exception &err)                                                 \
{                                                                           \
    NODE_LOG(error)<<err.what();										    \
}catch(...)                                                                 \
{                                                                           \
    NODE_LOG(error)<<"Unknown exception";                                   \
}



#define NODE_DEFAULT_CONSTRUCTOR_IMPL(NodeName) \
NodeName::NodeName():Node()                     \
{                                               \
    nodeName = #NodeName;                       \
    treeName = nodeName;						\
	fullTreeName = treeName;					\
}												\
REGISTERCLASS(NodeName)


namespace EagleLib
{
    class NodeManager;
    class Node;

	enum NodeType
	{
		eVirtual		= 1,	/* This is a virtual node, it should only be inherited */
		eGPU			= 2,	/* This node processes on the GPU, if this flag isn't set it processes on the CPU*/
		eImg			= 4,	/* This node processes images */
		ePtCloud		= 8,	/* This node processes point cloud data */
		eProcessing		= 16,	/* Calling the doProcess function actually does something */
		eFunctor		= 32,   /* Calling doProcess doesn't do anything, instead this node presents a function to be used in another node */
		eObj			= 64,	/* Calling doProcess doesn't do anything, instead this node presents a object that can be used in another node */
		eOneShot		= 128,	/* Calling doProcess does something, but should only be called once.  Maybe as a setup? */
		eSource         = 256,  /* this node generates data*/
		eSink           = 512   /* This node accepts and saves data */
    };
    enum Verbosity
    {
        Profiling = 0,
        Status = 1,
        Warning = 2,
        Error = 3,
        Critical = 4
    };
	

    class CV_EXPORTS Node: public TInterface<IID_NodeObject, IObject>, public IObjectNotifiable
    {
    public:
        static Verbosity  debug_verbosity;
        typedef shared_ptr<Node> Ptr;
		Node();
		virtual ~Node();
        /**
         * @brief process Gets called by processing threads and parent nodes.  Process should be left alone because
         * @brief it is where all of the exception handling occurs
         * @param img input image
         * @param steam input stream
         * @return output image
         */
        virtual cv::cuda::GpuMat        process(cv::cuda::GpuMat& img, cv::cuda::Stream& steam = cv::cuda::Stream::Null());
		virtual void					process(cv::InputArray in, cv::OutputArray out);
        /**
         * @brief doProcess this is the most used node and where the bulk of the work is performed.
         * @param img input image
         * @param stream input stream
         * @return output image
         */
        virtual cv::cuda::GpuMat		doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		virtual void					doProcess(cv::cuda::GpuMat& img, boost::promise<cv::cuda::GpuMat>& retVal);
		virtual void					doProcess(cv::InputArray in, boost::promise<cv::OutputArray>& retVal);
		virtual void					doProcess(cv::InputArray in, cv::OutputArray out);

		virtual void					reset();

        /**
         * @brief getName depricated?  Idea was to recursively go through parent nodes and rebuild my tree name, useful I guess once
         * @brief node swapping and moving is implemented
         * @return
         */
		std::string						getName() const;
        /**
         * @brief getTreeName depricated?
         * @return
         */
		std::string						getTreeName() const;
        /**
         * @brief getParent returns a pointer to the parent node... depricated?
         * @return
         */
        Node *getParent();
        /**
         * @brief getInputs [DEPRICATED]
         */
		virtual void					getInputs();
        /**
         * @brief log internally used to log node status, warnings, and errors
         * @param level
         * @param msg
         */
		//virtual void                    log(boost::log::trivial::severity_level level, const std::string& msg);

        virtual void updateParent();



		// ****************************************************************************************************************
		//
		//									Display functions
		//
		// ****************************************************************************************************************
		// Register a function for displaying CPU images
         virtual void registerDisplayCallback(boost::function<void(cv::Mat, Node*)>& f);
		// Register a function for displaying GPU images
         virtual void registerDisplayCallback(boost::function<void(cv::cuda::GpuMat, Node*)>& f);
		// Spawn an external display just for this node, with name = treeName
		 virtual void spawnDisplay();
		// Kill any spawned external displays
		 virtual void killDisplay();

		// ****************************************************************************************************************
		//
        //									Child adding and deleting
		//
		// ****************************************************************************************************************
        virtual Node::Ptr               addChild(Node* child);
        virtual Node::Ptr               addChild(Node::Ptr child);
        virtual Node::Ptr               getChild(const std::string& treeName);
        virtual Node::Ptr               getChild(const int& index);
        template<typename T> T* getChild(int index)
        {
            return dynamic_cast<T*>(index);
        }
        template<typename T> T* getChild(const std::string& name)
        {
            return dynamic_cast<T*>(name);
		}
        virtual Node*					getChildRecursive(std::string treeName_);
        virtual void					removeChild(const std::string& name);
        virtual void					removeChild(EagleLib::Node::Ptr node);
        virtual void					removeChild(int idx);
        virtual void                    swapChildren(int idx1, int idx2);
        virtual void                    swapChildren(const std::string& name1, const std::string& name2);
        virtual void                    swapChildren(Node::Ptr child1, Node::Ptr child2);
        virtual std::vector<Node*>  getNodesInScope();
        virtual Node *getNodeInScope(const std::string& name);
        virtual void getNodesInScope(std::vector<Node*>& nodes);
		
		// ****************************************************************************************************************
		//
		//									Name and accessing functions
		//
		// ****************************************************************************************************************
        /**
         * @brief setTreeName
         * @param name
         */
		virtual void setTreeName(const std::string& name);
        /**
         * @brief setFullTreeName
         * @param name
         */
		virtual void setFullTreeName(const std::string& name);
        /**
         * @brief setParent
         * @param parent
         */
        virtual void setParent(Node *parent);
        /**
         * @brief updateObject [DEPRICATED]
         * @param ptr
         */
        virtual void updateObject(IObject* ptr);
		


        /*******************************************************************************************************************
		//
		//									Parameter updating, getting and searching
		//

        Example usage:
            As a user updated control parameter:

                addParameter<int>("Integer Control Parameter", 0, Parameter::Control, "Tooltip for this parameter", false);
                - name is the name used in the property tree to identify this parameter, cannot contain / or : markings
                -- treeName will automatically be updated to be: parentNode->treeName:thisParameter->name
                - data is the the data that this parameter will store.
                - Type is the type of paramter
                - toolTip is a string used to describe the parameter to the user
                - ownsData_ is only used in the case where a pointer is passed in as data, if it is true it will automatically
                  delete the pointer when no more references to the paramter exist


            As an input
            addParameter<int*>("Integer Control Parameter", nullptr, Parameter::Input, "Tooltip", false);
			Integer Control Parameter will be the name 


        // ****************************************************************************************************************/
        // Find suitable input parameters
        /**
         * @brief listInputs
         * @return
         */
		virtual std::vector<std::string> listInputs();
        /**
         * @brief listParameters
         * @return
         */
		virtual std::vector<std::string>	 listParameters();
        /**
         * @brief findType
         * @param param
         * @return
         */
        virtual std::vector<std::string> findType(Parameters::Parameter::Ptr param);
        /**
         * @brief findType
         * @param typeInfo
         * @return
         */
        virtual std::vector<std::string> findType(Loki::TypeInfo typeInfo);
        /**
         * @brief findType
         * @param typeInfo
         * @param nodes
         * @return
         */
        virtual std::vector<std::string> findType(Loki::TypeInfo& typeInfo, std::vector<Node*>& nodes);
        /**
         * @brief findType
         * @param param
         * @param nodes
         * @return
         */
        virtual std::vector<std::string> findType(Parameters::Parameter::Ptr param, std::vector<Node *> &nodes);
        /**
         * @brief findCompatibleInputs
         * @return
         */
		virtual std::vector<std::vector<std::string>> findCompatibleInputs();
        /**
         * @brief findCompatibleInputs
         * @param paramName
         * @return
         */
        std::vector<std::string> findCompatibleInputs(const std::string& paramName);
        /**
         * @brief findCompatibleInputs
         * @param paramIdx
         * @return
         */
        std::vector<std::string> findCompatibleInputs(int paramIdx);
        /**
         * @brief findCompatibleInputs
         * @param param
         * @return
         */
        std::vector<std::string> findCompatibleInputs(Parameters::Parameter::Ptr param);
        /**
         * @brief setInputParameter
         * @param sourceName
         * @param inputName
         */
		std::vector<std::string> findCompatibleInputs(Loki::TypeInfo& type);

		std::vector<std::string> findCompatibleInputs(Parameters::InputParameter::Ptr param);

        virtual void setInputParameter(const std::string& sourceName, const std::string& inputName);
        /**
         * @brief setInputParameter
         * @param sourceName
         * @param inputIdx
         */
        virtual void setInputParameter(const std::string& sourceName, int inputIdx);
        /**
         * @brief updateInputParameters
         */
		virtual void updateInputParameters();

        /**
         * @brief getParameter
         * @param idx
         * @return
         */
		virtual Parameters::Parameter::Ptr getParameter(int idx);

        /**
         * @brief getParameter
         * @param name
         * @return
         */
		virtual Parameters::Parameter::Ptr getParameter(const std::string& name);
		
        /**
         * @brief addParameter
         * @param name
         * @param data
         * @param type_
         * @param toolTip_
         * @param ownsData_
         * @return
         */
		/*template<typename T, typename Enable = void> size_t addParameter(const std::string& name,
			const T& data, Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control, const std::string& toolTip_ = std::string(), bool ownsData = false)
		{
			return 0;
		}*/

		void RegisterParameterCallback(int idx, boost::function<void(void)> callback);
		void RegisterParameterCallback(const std::string& name, boost::function<void(void)> callback);

		template<typename T> size_t registerParameter(
			const std::string& name,
			T* data,
			Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control,
			const std::string& toolTip_ = std::string(),
			bool ownsData = false)
		{
			parameters.push_back(typename Parameters::TypedParameterPtr<T>::Ptr(new Parameters::TypedParameterPtr<T>(name, data, type_, toolTip_)));
			parameters[parameters.size() - 1]->SetTreeRoot(fullTreeName);
            onParameterAdded(this);
			return parameters.size() - 1;
		}

		
		template<typename T> size_t addParameter(const std::string& name,
				const T& data,
				Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control,
				const std::string& toolTip_ = std::string(), 
				bool ownsData = false/*, typename std::enable_if<!std::is_pointer<T>::value, void>::type* dummy_enable = nullptr*/)
		{
            parameters.push_back(typename Parameters::TypedParameter<T>::Ptr(new Parameters::TypedParameter<T>(name, data, type_, toolTip_)));
			parameters[parameters.size() - 1]->SetTreeRoot(fullTreeName);
            onParameterAdded(this);
			return parameters.size() - 1;
		}

        /**
          * @brief addInputParameter is used to define an input parameter for a node
          * @param name is the internal name of the input parameter
          * @param toolTip_ is the tooltip to be displayed on the UI
          * @return the index of the parameter
          */
         template<typename T> size_t
			 addInputParameter(const std::string& name, const std::string& toolTip_ = std::string(), const boost::function<bool(Parameters::Parameter*)>& qualifier_ = boost::function<bool(Parameters::Parameter*)>())
		{
                parameters.push_back(typename Parameters::TypedInputParameter<T>::Ptr(new Parameters::TypedInputParameter<T>(name, toolTip_, qualifier_)));
				parameters[parameters.size() - 1]->SetTreeRoot(fullTreeName);
            onParameterAdded(this);
			return parameters.size() - 1;
		}

        template<typename T> bool
            updateInputQualifier(const std::string& name, const boost::function<bool(Parameters::Parameter::Ptr&)>& qualifier_)
        {
            auto param = getParameter(name);
			if (param && param->type & Parameters::Parameter::Input)
			{
				Parameters::InputParameter* inputParam = dynamic_cast<Parameters::InputParameter*>(param.get());
				if (inputParam)
				{
					inputParam->SetQualifier(qualifier_);
					return true;
				}
			}
			return false;
        }
        template<typename T> bool
            updateInputQualifier(int idx, const boost::function<bool(Parameters::Parameter*)>& qualifier_)
        {
            auto param = getParameter<T>(idx);
            if(param && param->type & Parameters::Parameter::Input)
            {
				Parameters::InputParameter* inputParam = dynamic_cast<Parameters::InputParameter*>(param.get());
                if(inputParam)
                {
					inputParam->SetQualifier(qualifier_);
                    return true;
                }
            }
            return false;
        }

		template<typename T> bool updateParameterPtr(
												const std::string& name, 
												T* data, 
												Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control,
												const std::string& toolTip_ = std::string(), 
												const bool ownsData_ = false)
		{
			typename Parameters::ITypedParameter<T>::Ptr param;
			try
			{
				param = getParameter<T>(name);
			}
			catch (cv::Exception &e)
			{
				e.what();
				NODE_LOG(debug) << name << " doesn't exist, adding";
				return registerParameter<T>(name, data, type_, toolTip_, ownsData_);
			}
			param->UpdateData(data);
            return true;
			
		}


        template<typename T> bool updateParameter(const std::string& name,
												  const T& data,
												  Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control,
												  const std::string& toolTip_ = std::string(),
												  const bool& ownsData_ = false)
		{
            typename Parameters::ITypedParameter<T>::Ptr param;
            try
            {
                param = getParameter<T>(name);
            }catch(cv::Exception &e)
            {
				e.what();
				NODE_LOG(debug) << "Parameter named \"" << name << "\" with type " << Loki::TypeInfo(typeid(T)).name() << " doesn't exist, adding";
                return addParameter<T>(name, data, type_, toolTip_, ownsData_);
            }
			if (type_ != Parameters::Parameter::None)
				param->type = type_;
			if (toolTip_.size() > 0)
				param->SetTooltip(toolTip_);

			param->UpdateData(data);
			return true;
        }

        template<typename T> bool updateParameter(size_t idx,
												  const T data,
												  const std::string& name = std::string(),
												  const std::string quickHelp = std::string(),
												  Parameters::Parameter::ParameterType type_ = Parameters::Parameter::Control)
		{
			if (idx > parameters.size() || idx < 0)
				return false;
			auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<T>>(parameters[idx]);
			if (param == NULL)
				return false;

			if (name.size() > 0)
				param->SetName( name );
			if (type_ != Parameters::Parameter::None)
				param->type = type_;
			if (quickHelp.size() > 0)
				param->SetTooltip(quickHelp);
			param->UpdateData(data);
			return true;
		}
		


		template<typename T> typename Parameters::ITypedParameter<T>::Ptr
			getParameter(std::string name)
		{
			auto param =  getParameter(name);
			if (param == nullptr)
            {
                throw cv::Exception(0, "Failed to get parameter by name " + name, __FUNCTION__, __FILE__, __LINE__);
                return typename Parameters::ITypedParameter<T>::Ptr();
            }
			auto typedParam = std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
            if(typedParam == nullptr)
                throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
                    TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + param->GetTypeInfo().name(), __FUNCTION__, __FILE__, __LINE__);
			
            return typedParam;
		}

		template<typename T> typename Parameters::ITypedParameter<T>::Ptr getParameter(int idx)
		{
            auto param = getParameter(idx);
            if(param == nullptr)
                throw cv::Exception(0, "Failed to get parameter by index " + boost::lexical_cast<std::string>(idx), __FUNCTION__, __FILE__, __LINE__);

            auto typedParam = std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(param);
            if(typedParam == nullptr)
                throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
                    TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + param->GetTypeInfo().name(), __FUNCTION__, __FILE__, __LINE__);
            return typedParam;
		}


		/*!
		*  \brief findInputs recursively finds any compatible inputs wrt the templated desired type.
		*  \brief usage includes finding all output images
		*  \param output is a vector of the output parameters including a list of the names of where they are from
		*/
		template<typename T> void
			findInputs(std::vector<std::string>& nodeNames, std::vector< typename Parameters::ITypedParameter<T>::Ptr>& parameterPtrs, int hops = 10000)
		{
			if (hops < 0)
				return;
			for (int i = 0; i < parameters.size(); ++i)
			{
				if (parameters[i]->type & Parameters::Parameter::Output) // Can't use someone's input or control parameter, that would be naughty
					if (std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(parameters[i]))
					{
						nodeNames.push_back(treeName);
						parameterPtrs.push_back(std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(parameters[i]));
					}
			}
			// Recursively check children for any available output parameters that match the input signature
			/*for(int i = 0; i < children.size(); ++i)
			children[i]->findInputs<T>(nodeNames, parameterPtrs, hops - 1);*/
			for (auto itr = children.begin(); itr != children.end(); ++itr)
				itr->findInputs<T>(nodeNames, parameterPtrs, hops - 1);

			return;
		}


       

        // ****************************************************************************************************************
        //
        //									Dynamic reloading and persistence
        //
        // ****************************************************************************************************************

        /**
         * @brief swap is supposed to swap positions of this node and other node.  This would swap it into the same place
         * @brief wrt other's parent, and swap all children.  This would not do anything to the parameters.  Not implemented
         * @brief and tested yet.
         * @param other
         * @return
         */
        virtual Node *swap(Node *other);

        virtual void Init(bool firstInit = true);
        /**
         * @brief Init [DEPRICATED], would be used to load the configuration of this node froma  file
         * @param configFile
         */
        virtual void Init(const std::string& configFile);
        /**
         * @brief Init is used to reload a node based on a saved .yml configuration.
         * @brief base implementation reloads children, parameters and name information.
         * @brief override to extend to include other node specific persistence.
         * @brief Needs to match with the Serialize(cv::FileStorage& fs) function.
         * @param configNode is the fileNode representing the configuration of this node
         */
        virtual void Init(const cv::FileNode& configNode);
        /**
         * @brief Serialize serializes in and out the variables of a node during runtime object swapping.
         * @brief The default implementation handles serializing name parameters, children and parameters.
         * @param pSerializer is the serializer object that stores variables during swap
         */
        virtual void Serialize(ISimpleSerializer *pSerializer);
        /**
         * @brief Serialize [DEPRICATED] initial intention was to use this for persistence
         * @param fs
         */
        virtual void Serialize(cv::FileStorage& fs);
        /**
         * @brief SkipEmpty is used to flag if a node's doProcess function should be called on an empty input img.
         * @brief by default, all nodes just skip processing on an empty input.  For file loading nodes, this isn't desirable
         * @brief so for any nodes that generate an output, they override this function to return false.
         * @return true if this node should skip empty images, false otherwise.
         */
        virtual bool SkipEmpty() const;

        // ****************************************************************************************************************
        //
        //									Members
        //
        // ****************************************************************************************************************

        // Used for logging / UI updating when an error occurs.
		boost::function<void(boost::log::trivial::severity_level, const std::string&, Node*)>         messageCallback;
        // Depricated
        boost::function<int(std::vector<std::string>)>						inputSelector;
        // Used to signal if an update is required.  IE if performing static image analysis, if no parameters of a node
        // are updated, there is no reason to re-process the image with the same parameters.
        // Just need to figure out the best way of implementing it.
        boost::function<void(Node*)>                                        onUpdate;
        // Vector of child nodes
        std::vector<Node::Ptr>                                              children;

		// Constant name that describes the node ie: Sobel
        std::string															nodeName;
		// Name as placed in the tree ie: RootNode/SerialStack/Sobel-1
        std::string															fullTreeName;       
		// Name as it is stored in the children map, should be unique at this point in the tree. IE: Sobel-1
        std::string															treeName;
        // Vector of parameters for this node
        std::vector< Parameters::Parameter::Ptr >							parameters;
        // These can be used to for user defined UI displaying of images.  IE if you havea  custom widget for displaying
        // nodes, you can plug that in here.
        boost::function<void(cv::Mat, Node*)>								cpuDisplayCallback;
        boost::function<void(cv::cuda::GpuMat, Node*)>						gpuDisplayCallback;
        // This is depricated since we now use the UIThreadCallback singleton for posting functions to a queue for processing
        boost::function<void(boost::function<cv::Mat()>, Node*)>            uiThreadCallback;
        boost::function<void(boost::function<void()>, Node*)>               d_uiThreadCallback;

        /* If true, draw results onto the image being processed, hardly ever used */
        bool																drawResults;
		/* True if spawnDisplay has been called, in which case results should be drawn and displayed on a window with the name treeName */
		bool																externalDisplay;
        // Toggling this disables a node's doProcess code from ever being called
        bool                                                                enabled;

        bool                                                                profile;

        double                                                              processingTime;
        // Mutex for blocking processing of a node during parameter update
        boost::recursive_mutex                                              mtx;
        boost::signals2::signal<void(Node*)>								onParameterAdded;
		
        std::vector<std::pair<clock_t, int>> timings;
		NodeType															nodeType;
	protected:
		static boost::signals2::signal<void(void)>							resetSignal;
    private:
        friend class NodeManager;
        // Depricated, I think
        ObjectId                                                            m_OID;
        // Pointer to parent node
        Node*                                                               parent;
        boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::rolling_mean> > averageFrameTime;
        ConstBuffer<cv::cuda::GpuMat>                                       childResults;
		
		boost::signals2::connection											resetConnection;
		std::vector<boost::signals2::connection>							callbackConnections;
        boost::posix_time::ptime lastStatusTime;
        boost::posix_time::ptime lastWarningTime;
        boost::posix_time::ptime lastErrorTime;
        boost::posix_time::ptime lastCriticalTime;
        std::string lastStatusMsg;
        std::string lastWarningMsg;
        std::string lastErrorMsg;
        std::string lastCriticalMsg;
    };

}
