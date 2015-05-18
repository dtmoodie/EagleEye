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

#include "../EagleLib.h"
#include "../Manager.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/signals2.hpp>
#include <boost/thread/future.hpp> 
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/mpl/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <vector>
#include <list>
#include <map>
#include <type_traits>
#include <boost/filesystem.hpp>
#include "../LokiTypeInfo.h"
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include "../Parameters.h"


#ifdef HAVE_PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/pcl_macros.h>
#else
#pragma warning Not building with PCL support
#endif

#define TIME timings.push_back(clock());
using namespace boost::multi_index;

#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "../../RuntimeObjectSystem/ObjectInterface.h"
#include "../../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "../../RuntimeObjectSystem/IObject.h"

#ifdef _MSC_VER


#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core")

#endif



#define NODE_DEFAULT_CONSTRUCTOR_IMPL(NodeName) \
NodeName::NodeName():Node()                     \
{                                               \
    nodeName = #NodeName;                       \
    treeName = nodeName;						\
	fullTreeName = treeName;					\
}												\
REGISTERCLASS(NodeName)

#define EAGLE_TRY_WARNING(FunctionCall)                                 \
    try{                                                                \
    (FunctionCall)                                                      \
    }catch(cv::Exception &e){                                           \
        if(warningCallback)                                             \
            warningCallback(std::string(__FUNCTION__) + e.what());                   \
    }catch(std::exception &e){                                          \
        if(warningCallback)                                             \
            warningCallback(std::string(__FUNCTION__) + e.what());                   \
    }

#define EAGLE_TRY_ERROR(FunctionCall)                                   \
    try{                                                                \
    FunctionCall														\
    }catch(cv::Exception &e){                                           \
        if(errorCallback)                                               \
            errorCallback(std::string(__FUNCTION__) + e.what());                     \
    }catch(std::exception &e){                                          \
        if(errorCallback)                                               \
            errorCallback(std::string(__FUNCTION__) + e.what());                     \
    }
#define EAGLE_ERROR_CHECK_RESULT(FunctionCall, DesiredResult)                 \
    if(FunctionCall != DesiredResult){                                  \
        if(errorCallback)                                               \
            errorCallback(std::string(__FUNCION__ + " " + #FunctionCall + " != " #DesiredResult);                                                                \
    }
        
#define AddParameter(Parameter, ...)        addParameter(#Parameter, Parameter, __VA_ARGS__)
#define AddOutputParameter(Parameter, ...)  addParameter(#Parameter, &Parameter, Parameter::Output, __VA_ARGS__);


namespace EagleLib
{
    class NodeManager;
    class Node;



	enum NodeType
	{
		eVirtual		= 0,	/* This is a virtual node, it should only be inherited */
		eGPU			= 1,	/* This node processes on the GPU, if this flag isn't set it processes on the CPU*/
		eImg			= 2,	/* This node processes images */
		ePtCloud		= 4,	/* This node processes point cloud data */
		eProcessing		= 8,	/* Calling the doProcess function actually does something */
		eFunctor		= 16,   /* Calling doProcess doesn't do anything, instead this node presents a function to be used in another node */
		eObj			= 32,	/* Calling doProcess doesn't do anything, instead this node presents a object that can be used in another node */
		eOneShot		= 64	/* Calling doProcess does something, but should only be called once.  Maybe as a setup? */
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


        //static void registerType(const std::string& name, NodeFactory* factory);
		

		Node();
		virtual ~Node();
        
        virtual cv::cuda::GpuMat        process(cv::cuda::GpuMat& img, cv::cuda::Stream& steam = cv::cuda::Stream::Null());
		virtual void					process(cv::InputArray in, cv::OutputArray out);
		// Processing functions, these actually do the work of the node
        virtual cv::cuda::GpuMat		doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
		virtual void					doProcess(cv::cuda::GpuMat& img, boost::promise<cv::cuda::GpuMat>& retVal);
		virtual void					doProcess(cv::InputArray in, boost::promise<cv::OutputArray>& retVal);
		virtual void					doProcess(cv::InputArray in, cv::OutputArray out);

        // Finds name in tree hierarchy, updates tree name and returns it
		std::string						getName() const;
		std::string						getTreeName() const;
        Node *getParent();
		// Searches nearby nodes for possible valid inputs for each input parameter
		virtual void					getInputs();
        virtual void                    log(Verbosity level, const std::string& msg);

        struct NodeInfo
        {
            int index;
            std::string treeName;
            std::string nodeName;
            ObjectId id;
        };



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
		virtual void setTreeName(const std::string& name);
		virtual void setFullTreeName(const std::string& name);
        virtual void setParent(Node *parent);
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
		virtual std::vector<std::string> listInputs();
		virtual std::vector<std::string>	 listParameters();
        virtual std::vector<std::string> findType(Parameter::Ptr param);
        virtual std::vector<std::string> findType(Loki::TypeInfo& typeInfo);
        virtual std::vector<std::string> findType(Loki::TypeInfo& typeInfo, std::vector<Node*>& nodes);
        virtual std::vector<std::string> findType(Parameter::Ptr param, std::vector<Node *> &nodes);
		virtual std::vector<std::vector<std::string>> findCompatibleInputs();
        std::vector<std::string> findCompatibleInputs(const std::string& paramName);
        std::vector<std::string> findCompatibleInputs(int paramIdx);
        std::vector<std::string> findCompatibleInputs(Parameter::Ptr param);
        virtual void setInputParameter(const std::string& sourceName, const std::string& inputName);
        virtual void setInputParameter(const std::string& sourceName, int inputIdx);
		virtual void updateInputParameters();
		virtual boost::shared_ptr<Parameter> getParameter(int idx);
		virtual boost::shared_ptr<Parameter> getParameter(const std::string& name);
		
		template<typename T> size_t
			addParameter(const std::string& name,
						const T& data,
						Parameter::ParamType type_ = Parameter::Control,
						const std::string& toolTip_ = std::string(),
						const bool& ownsData_ = false)
		{
			if(std::is_pointer<T>::value)
				parameters.push_back(boost::shared_ptr< TypedParameter<T> >(new TypedParameter<T>(name, data, type_ + Parameter::NotifyOnRecompile, toolTip_, ownsData_)));
			else
					parameters.push_back(boost::shared_ptr< TypedParameter<T> >(new TypedParameter<T>(name, data, type_, toolTip_, ownsData_)));
            parameters[parameters.size() - 1]->treeName = fullTreeName + ":" + parameters[parameters.size() - 1]->name;
            (*onParameterAdded)();
			return parameters.size() - 1;
		}
        template<typename T> size_t
            addParameter(const std::string& name,
                        T& data,
                        Parameter::ParamType type_ = Parameter::Control,
                        const std::string& toolTip_ = std::string(),
                        const bool& ownsData_ = false)
        {
            if(std::is_pointer<T>::value)
                parameters.push_back(boost::shared_ptr< TypedParameter<T> >(new TypedParameter<T>(name, data, type_ + Parameter::NotifyOnRecompile, toolTip_, ownsData_)));
            else
                    parameters.push_back(boost::shared_ptr< TypedParameter<T> >(new TypedParameter<T>(name, data, type_, toolTip_, ownsData_)));
            parameters[parameters.size() - 1]->treeName = fullTreeName + ":" + parameters[parameters.size() - 1]->name;
            (*onParameterAdded)();
            return parameters.size() - 1;
        }


		template<typename T> size_t
			addInputParameter(const std::string& name, const std::string& toolTip_ = std::string())
		{
			parameters.push_back(boost::shared_ptr< InputParameter<T> >(new InputParameter<T>(name, toolTip_)));
            parameters[parameters.size() - 1]->treeName = fullTreeName + ":" + parameters[parameters.size() - 1]->name;
			return parameters.size() - 1;
		}


		template<typename T> bool
			updateParameter(const std::string& name,
							const T& data,
							Parameter::ParamType type_ = Parameter::Control,
							const std::string& toolTip_ = std::string(),
							const bool& ownsData_ = false)
		{
			auto param = getParameter<T>(name);
			if (param == NULL)
				return addParameter(name, data, type_, toolTip_, ownsData_);
			param->data = data;
			if (type_ != Parameter::None)
				param->type = type_;
			if (toolTip_.size() > 0)
				param->toolTip = toolTip_;
			param->changed = true;
            param->onUpdate();
			return true;
		}
        template<typename T> bool
            updateParameter(const std::string& name,
                            T& data,
                            Parameter::ParamType type_ = Parameter::Control,
                            const std::string& toolTip_ = std::string(),
                            const bool& ownsData_ = false)
        {
            auto param = getParameter<T>(name);
            if (param == NULL)
                return addParameter(name, data, type_, toolTip_, ownsData_);
            param->data = data;
            if (type_ != Parameter::None)
                param->type = type_;
            if (toolTip_.size() > 0)
                param->toolTip = toolTip_;
            param->changed = true;
            param->onUpdate();
            return true;
        }

		template<typename T> bool
			updateParameter(int idx,
							T data,
							const std::string& name = std::string(),
							const std::string quickHelp = std::string(),
							Parameter::ParamType type_ = Parameter::None)
		{
			if (idx > parameters.size() || idx < 0)
				return false;
			typename TypedParameter<T>::Ptr param = boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(parameters[idx]);
			if (param == NULL)
				return false;

			if (name.size() > 0)
				param->name = name;
			if (type_ != Parameter::None)
				param->type = type_;
			if (quickHelp.size() > 0)
				param->toolTip = quickHelp;
            param->data = data;
            param->changed = true;
            param->onUpdate();
			return true;
		}

		// Recursively searchs for a parameter based on name
		template<typename T> boost::shared_ptr< TypedParameter<T> >
			getParameterRecursive(std::string name, int depth)
		{
			if (depth < 0)
				return boost::shared_ptr < TypedParameter<T> >();
			for (int i = 0; i < parameters.size(); ++i)
			{
				if (parameters[i]->name == name)
					return boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(parameters[i]);
			}
			// Parameter doesn't exist in this scope, we must go deeper
			for (auto itr = children.begin(); itr != children.end(); ++itr)
			{
				boost::shared_ptr< TypedParameter<T> > param = itr->getParameterRecursive<T>(name, depth - 1);
				if (param)
					return param;
			}
			return boost::shared_ptr< TypedParameter<T> >();
		}
		template<typename T> boost::shared_ptr< TypedParameter<T> >
			getParameter(std::string name)
		{
			auto param =  getParameter(name);
			if (param == nullptr)
				return boost::shared_ptr<TypedParameter<T>>();


			
			return boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(param);
		}

		template<typename T> boost::shared_ptr< TypedParameter<T> >
			getParameter(int idx)
		{
            Parameter::Ptr param = getParameter(idx);
            if(param == nullptr)
                throw cv::Exception(0, "Failed to get parameter by index " + boost::lexical_cast<std::string>(idx), __FUNCTION__, __FILE__, __LINE__);
            boost::shared_ptr< TypedParameter<T> > typedParam = boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(param);
            if(typedParam == nullptr)
#ifdef _MSC_VER
				throw cv::Exception(0, std::string("Failed to cast parameter to the appropriate type, requested type: ") +  std::string(typeid(T).name()) + std::string(" parameter actual type: ") + std::string(  param->typeInfo.name()), __FUNCTION__, __FILE__, __LINE__);
#else

			throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
                TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + TypeInfo::demangle(param->typeInfo.name()), __FUNCTION__, __FILE__, __LINE__);
#endif
			
            return typedParam;
		}

					//
		bool
			subParameterExists(std::string name)
		{

			return false;
		}

					// Check to see if a sub parameter is of a certain type
		template<typename T> bool
			checkSubParameterType(std::string name)
		{

		}
					// Get's a pointer to a sub parameter based on the name of the sub parameter
		template<typename T> boost::shared_ptr< TypedParameter<T> >
			getSubParameter(std::string name)
		{

			return boost::shared_ptr< TypedParameter<T> >(); // Return a NULL pointer
		}

		/*!
		*  \brief findInputs recursively finds any compatible inputs wrt the templated desired type.
		*  \brief usage includes finding all output images
		*  \param output is a vector of the output parameters including a list of the names of where they are from
		*/
		template<typename T> void
			findInputs(std::vector<std::string>& nodeNames, std::vector< boost::shared_ptr< TypedParameter<T> > >& parameterPtrs, int hops = 10000)
		{
			if (hops < 0)
				return;
			for (int i = 0; i < parameters.size(); ++i)
			{
				if (parameters[i]->type & Parameter::Output) // Can't use someone's input or control parameter, that would be naughty
					if (boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(parameters[i]))
					{
						nodeNames.push_back(treeName);
						parameterPtrs.push_back(boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(parameters[i]));
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


        virtual Node *swap(Node *other);

        virtual void Init(bool firstInit = true);
        virtual void Init(const std::string& configFile);
        virtual void Init(const cv::FileNode& configNode);
        virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual void Serialize(cv::FileStorage& fs);

        virtual bool SkipEmpty() const;

        // ****************************************************************************************************************
        //
        //									Members
        //
        // ****************************************************************************************************************


        boost::function<void(Verbosity, const std::string&, Node*)>         messageCallback;
		// Function for setting input parameters
        boost::function<int(std::vector<std::string>)>						inputSelector;
        boost::function<void(Node*)>                                        onUpdate;
        std::vector<Node::Ptr>                                              children;

		// Constant name that describes the node ie: Sobel
        std::string															nodeName;
		// Name as placed in the tree ie: RootNode/SerialStack/Sobel-1
        std::string															fullTreeName;       
		// Name as it is stored in the children map, should be unique at this point in the tree. IE: Sobel-1
        std::string															treeName;
        std::vector< boost::shared_ptr< Parameter > >						parameters;
        boost::function<void(cv::Mat, Node*)>								cpuDisplayCallback;
        boost::function<void(cv::cuda::GpuMat, Node*)>						gpuDisplayCallback;
        // This is a function that operates on the UI thread, on a cv::Mat for display.
        // This can be used for
        boost::function<void(boost::function<cv::Mat()>, Node*)>            uiThreadCallback;
        boost::function<void(boost::function<void()>, Node*)>               d_uiThreadCallback;
		/* If true, draw results onto the image being processed */
        bool																drawResults;
		/* True if spawnDisplay has been called, in which case results should be drawn and displayed on a window with the name treeName */
		bool																externalDisplay;
        bool                                                                enabled;
        bool                                                                profile;
        double                                                              processingTime;
        boost::recursive_mutex                                              mtx;
        boost::shared_ptr<boost::signals2::signal<void(void)>>              onParameterAdded;
        std::vector<clock_t> timings;
    private:
        friend class NodeManager;
        ObjectId                                                            m_OID;
        Node*                                                               parent;
        boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::rolling_mean> > averageFrameTime;
        ConstBuffer<cv::cuda::GpuMat>                                       childResults;
    };
    
    class CV_EXPORTS EventLoopNode: public Node
    {
    protected:
        boost::asio::io_service service;
    public:
        EventLoopNode();
        virtual cv::cuda::GpuMat process(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

}
