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


#include <opencv2/core/cuda.hpp>
//#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/signals2.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/filesystem.hpp>
#include <vector>
#include <list>
#include <map>
#include <type_traits>

#include "../LokiTypeInfo.h"
#include <boost/thread.hpp>
#include <Qualifiers.hpp>
#include "../Parameters.h"
#include <external_includes/cv_core.hpp>
#define TIME if(profile) timings.push_back(std::pair<clock_t, int>(clock(), __LINE__));

#include "RuntimeLinkLibrary.h"
#include "ObjectInterface.h"
#include "ObjectInterfacePerModule.h"
#include "IObject.h"

#ifdef _MSC_VER
#ifdef _DEBUG
	RUNTIME_COMPILER_LINKLIBRARY("EagleLib.lib")
#else
	RUNTIME_COMPILER_LINKLIBRARY("EagleLib.lib")
#endif
#else
	RUNTIME_COMPILER_LINKLIBRARY("-lEagleLib")
#endif

#define CATCH_MACRO                                                         \
}catch (boost::thread_resource_error& err)                                  \
{                                                                           \
    log(Error, err.what());                                                 \
}catch (boost::thread_interrupted& err)                                     \
{                                                                           \
    log(Error, "Thread interrupted");                                       \
    /* Needs to pass this back up to the chain to the processing thread.*/    \
    /* That way it knowns it needs to exit this thread */                     \
    throw err;                                                              \
}catch (boost::thread_exception& err)                                       \
{                                                                           \
    log(Error, err.what());                                                 \
}                                                                           \
catch (cv::Exception &err)                                                  \
{                                                                           \
    log(Error, err.what());                                                 \
}                                                                           \
catch (boost::exception &err)                                               \
{                                                                           \
    log(Error, "Boost error");                                              \
}catch(std::exception &err)                                                 \
{                                                                           \
    log(Error, err.what());                                                 \
}catch(...)                                                                 \
{                                                                           \
    log(Error, "Unknown exception");                                        \
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
        virtual void                    log(Verbosity level, const std::string& msg);
        /**
         * @brief The NodeInfo struct [DEPRICATED]
         */
        struct NodeInfo
        {
            int index;
            std::string treeName;
            std::string nodeName;
            ObjectId id;
        };

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
        virtual std::vector<std::string> findType(Parameter::Ptr param);
        /**
         * @brief findType
         * @param typeInfo
         * @return
         */
        virtual std::vector<std::string> findType(Loki::TypeInfo& typeInfo);
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
        virtual std::vector<std::string> findType(Parameter::Ptr param, std::vector<Node *> &nodes);
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
        std::vector<std::string> findCompatibleInputs(Parameter::Ptr param);
        /**
         * @brief setInputParameter
         * @param sourceName
         * @param inputName
         */
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
		virtual boost::shared_ptr<Parameter> getParameter(int idx);

        /**
         * @brief getParameter
         * @param name
         * @return
         */
		virtual boost::shared_ptr<Parameter> getParameter(const std::string& name);
		
        /**
         * @brief addParameter
         * @param name
         * @param data
         * @param type_
         * @param toolTip_
         * @param ownsData_
         * @return
         */
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
            if(onParameterAdded)
                (*onParameterAdded)();
			return parameters.size() - 1;
		}

        /**
         * @brief addParameter adds a parameter of specified type to the parameter list, mostly used by updateParameter, since update parameter checks for existence
         * @param name is the name for the parameter
         * @param data initialized data for the parameter
         * @param type_ is the parameter type, ie Control, Status, Input, Output
         * @param toolTip_ is the tooltip to be displayed on theuser interface for this parameter, ie units, ranges, etc
         * @param ownsData_ set to true if you want this parameter to try to delete the data upon parameter destruction
         * @return
         */
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
            if(onParameterAdded)
                (*onParameterAdded)();
            return parameters.size() - 1;
        }

        /**
          * @brief addInputParameter is used to define an input parameter for a node
          * @param name is the internal name of the input parameter
          * @param toolTip_ is the tooltip to be displayed on the UI
          * @return the index of the parameter
          */
         template<typename T> size_t
            addInputParameter(const std::string& name, const std::string& toolTip_ = std::string(), const boost::function<bool(const EagleLib::Parameter::Ptr&)>& qualifier_ = boost::function<bool(const EagleLib::Parameter::Ptr&)>())
		{
            parameters.push_back(boost::shared_ptr< InputParameter<T> >(new InputParameter<T>(name, toolTip_,qualifier_)));
            parameters[parameters.size() - 1]->treeName = fullTreeName + ":" + parameters[parameters.size() - 1]->name;
            if(onParameterAdded)
                (*onParameterAdded)();
			return parameters.size() - 1;
		}

        template<typename T> bool
            updateInputQualifier(const std::string& name, const boost::function<bool(const EagleLib::Parameter::Ptr&)>& qualifier_)
        {
            Parameter::Ptr param = getParameter(name);
            if(param)
            {
                typename EagleLib::InputParameter<T>::Ptr inputParam = boost::dynamic_pointer_cast<typename EagleLib::InputParameter<T>, typename EagleLib::Parameter>(param);
                if(inputParam)
                {
                    inputParam->qualifier = qualifier_;
                    inputParam->onUpdate();
                    return true;
                }
            }
            return false;
        }
        template<typename T> bool
            updateInputQualifier(int idx, const boost::function<bool(const EagleLib::Parameter::Ptr&)>& qualifier_)
        {
            Parameter::Ptr param = getParameter(idx);
            if(param)
            {
                typename EagleLib::InputParameter<T>::Ptr inputParam = boost::dynamic_pointer_cast<typename EagleLib::InputParameter<T>, typename EagleLib::Parameter>(param);
                if(inputParam)
                {
                    inputParam->qualifier = qualifier_;
                    inputParam->onUpdate();
                    return true;
                }
            }
            return false;
        }

        /**
          * @brief updateParameter is frequently used in nodes to add or update a parameter's value
          * @param name is the name of the parameter
          * @param data is the data that the parameter stores
          * @param type_ is the parameter type, ie Control, Status, Input, Output
          * @param toolTip_ is the tooltip to be displayed on the user interface for this parameter, IE units, ranges, etc
          * @param ownsData_ is used to flag if this parameter object owns the data being passed into it.
          * @param this is used if the parameter is a raw pointer, in which case on destruction of the parameter, it needs to be
          * @param deleted.
          * @return true on success, false on failure to add parameter.
          */

        template<typename T> bool
        updateParameter(const std::string& name,
                        const T& data,
                        Parameter::ParamType type_ = Parameter::Control,
                        const std::string& toolTip_ = std::string(),
                        const bool& ownsData_ = false)
		{
            typename TypedParameter<T>::Ptr param;
            try
            {
                param = getParameter<T>(name);
            }catch(cv::Exception &e)
            {
                return addParameter(name, data, type_, toolTip_, ownsData_);
            }
			param->data = data;
			if (type_ != Parameter::None)
				param->type = type_;
			if (toolTip_.size() > 0)
				param->toolTip = toolTip_;
			param->changed = true;
            param->onUpdate();
			return true;
		}
//        // Is this needed?  Will the above suffice?
//        template<typename T> bool
//            updateParameter(const std::string& name,
//                            T& data,
//                            Parameter::ParamType type_ = Parameter::Control,
//                            const std::string& toolTip_ = std::string(),
//                            const bool& ownsData_ = false)
//        {
//            auto param = getParameter<T>(name);
//            if (param == NULL)
//                return addParameter(name, data, type_, toolTip_, ownsData_);
//            param->data = data;
//            if (type_ != Parameter::None)
//                param->type = type_;
//            if (toolTip_.size() > 0)
//                param->toolTip = toolTip_;
//            param->changed = true;
//            param->onUpdate();
//            return true;
//        }
        /**
         * @brief updateParameter overload of the above accept accessing a paramter via index instead of name
         * @param idx index of parameter
         * @param data see above
         * @param name see above
         * @param quickHelp see above
         * @param type_ see above
         * @return see above
         */
        template<typename T> bool
            updateParameter(size_t idx,
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
            {
                throw cv::Exception(0, "Failed to get parameter by name " + name, __FUNCTION__, __FILE__, __LINE__);
				return boost::shared_ptr<TypedParameter<T>>();
            }
            boost::shared_ptr< TypedParameter<T> > typedParam = boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(param);
            if(typedParam == nullptr)
                throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
                    TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + TypeInfo::demangle(param->typeInfo.name()), __FUNCTION__, __FILE__, __LINE__);
			
            return typedParam;
		}

		template<typename T> boost::shared_ptr< TypedParameter<T> >
			getParameter(int idx)
		{
            Parameter::Ptr param = getParameter(idx);
            if(param == nullptr)
                throw cv::Exception(0, "Failed to get parameter by index " + boost::lexical_cast<std::string>(idx), __FUNCTION__, __FILE__, __LINE__);
            boost::shared_ptr< TypedParameter<T> > typedParam = boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(param);
            if(typedParam == nullptr)
                throw cv::Exception(0, "Failed to cast parameter to the appropriate type, requested type: " +
                    TypeInfo::demangle(typeid(T).name()) + " parameter actual type: " + TypeInfo::demangle(param->typeInfo.name()), __FUNCTION__, __FILE__, __LINE__);
            return typedParam;
		}

					//
		bool
			subParameterExists(std::string name)
		{

			return false;
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
        boost::function<void(Verbosity, const std::string&, Node*)>         messageCallback;
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
        std::vector< boost::shared_ptr< Parameter > >						parameters;
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
        boost::shared_ptr<boost::signals2::signal<void(void)>>              onParameterAdded;
        std::vector<std::pair<clock_t, int>> timings;
		NodeType															nodeType;
    private:
        friend class NodeManager;
        // Depricated, I think
        ObjectId                                                            m_OID;
        // Pointer to parent node
        Node*                                                               parent;
        boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::rolling_mean> > averageFrameTime;
        ConstBuffer<cv::cuda::GpuMat>                                       childResults;
    };

}
