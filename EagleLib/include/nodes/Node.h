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
*/

#include <opencv2/core.hpp>
#include <opencv2/cuda.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/signals2.hpp>
#include <boost/thread/future.hpp>





namespace EagleLib
{
    class CV_EXPORTS Parameter
    {
    public:
        enum ParamType
        {
            None = 0,
            Input = 1,
            Output = 2,
            User = 4,
            Local = 8,
            Shared = 16, // Shared accross all instances of this object
            Global = 32,
            State = 64 // State parameter to be read displayed
        };
        std::string name;
        std::string quickHelp;
        ParamType type;
    protected:
        Parameter(){}
        Parameter(const std::string& name_, const std::string& quickHelp_): name(name_), quickHelp(quickHelp_), type(None)  {}
        Parameter(const std::string& name_, const std::string& quickHelp_, ParamType type_): name(name_), quickHelp(quickHelp_),type(type_)  {}
        virtual ~Parameter(){}
    };

    template<typename T>
    class CV_EXPORTS TypedParameter: public Parameter
    {
    public:
        TypedParameter(){}
        TypedParameter(const std::string& name, const T& data_): Parameter(name," "){}
        TypedParameter(const std::string& name, const T& data_, ParamType type_ ): Parameter(name," "){}
        TypedParameter(const std::string& name, const std::string& quickHelp_): Parameter(name,quickHelp_){}
        TypedParameter(const std::string& name, const std::string& quickHelp_, ParamType type_): Parameter(name,quickHelp_, type_){}
        TypedParameter(const std::string& name, const std::string& quickHelp_, const T& data_): Parameter(name,quickHelp_), data(data_){}
        TypedParameter(const std::string& name, const std::string& quickHelp_, const T& data_, ParamType type_): Parameter(name,quickHelp_, type_), data(data_){}
        T& get();
        T data;
    };
    template<typename T>
    class CV_EXPORTS InputParameter: public TypedParameter<T>
    {
    public:
        InputParameter(){}
        InputParameter(const std::string& name, const T& data_): TypedParameter<T>(name, data_, Parameter::Input), changed(false){}
        InputParameter(const std::string& name, const T& data_, Parameter::ParamType type_ ): TypedParameter<T>(name,data_, type_ | Parameter::Input), changed(false){}
        InputParameter(const std::string& name, const std::string& quickHelp_): TypedParameter<T>(name,quickHelp_, Parameter::Input), changed(false){}
        InputParameter(const std::string& name, const std::string& quickHelp_, Parameter::ParamType type_): TypedParameter<T>(name,quickHelp_, type_ | Parameter::Input), changed(false){}
        InputParameter(const std::string& name, const std::string& quickHelp_, const T& data_): TypedParameter<T>(name,quickHelp_, data_), changed(false){}
        InputParameter(const std::string& name, const std::string& quickHelp_, const T& data_, Parameter::ParamType type_): TypedParameter<T>(name,quickHelp_, data_, type_ | Parameter::Input), changed(false){}
        bool changed;
    };
    template<typename T>
    class CV_EXPORTS OutputParameter: public TypedParameter<T>
    {
    public:
        OutputParameter(){}
        OutputParameter(const std::string& name, const T& data_): TypedParameter<T>(name, data_, Parameter::Output), changed(false){}
        OutputParameter(const std::string& name, const T& data_, Parameter::ParamType type_ ): TypedParameter<T>(name,data_, type_ | Parameter::Output), changed(false){}
        OutputParameter(const std::string& name, const std::string& quickHelp_): TypedParameter<T>(name,quickHelp_, Parameter::Output), changed(false){}
        OutputParameter(const std::string& name, const std::string& quickHelp_, Parameter::ParamType type_): TypedParameter<T>(name,quickHelp_, type_ | Parameter::Output), changed(false){}
        OutputParameter(const std::string& name, const std::string& quickHelp_, const T& data_): TypedParameter<T>(name,quickHelp_, data_, Parameter::Output), changed(false){}
        OutputParameter(const std::string& name, const std::string& quickHelp_, const T& data_, Parameter::ParamType type_): TypedParameter<T>(name,quickHelp_, data_, type_ | Parameter::Output), changed(false){}

        bool changed;
    };

    template<typename T>
    class CV_EXPORTS StaticTypedParameter: public Parameter
    {
    public:
        StaticTypedParameter() {}
        StaticTypedParameter(ParamType type_, const std::string& name, const std::string& quickHelp, const T& data_): Parameter(type_, name,quickHelp), data(data_){}
        T data;
    };
    template<typename T>
    class CV_EXPORTS ReferenceTypedParameter: public Parameter
    {
    public:
        ReferenceTypedParameter(){}
        ReferenceTypedParameter(ParamType type_, const std::string& name, const std::string& quickHelp, T& data_): Parameter(type_, name, quickHelp), data(data_){}
        T& data;
    };

    class CV_EXPORTS Node
    {
    public:
        Node();
        virtual ~Node();
        // Primary call to a node.  For simple operations this is the only thing that matters
        cv::cuda::GpuMat          process(cv::cuda::GpuMat& img);

        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);
        virtual void             doProcess(cv::cuda::GpuMat& img, boost::promise<cv::cuda::GpuMat>& retVal);
        virtual std::string      getName();

        /********************** Display stuff ****************************/
        // Registers a function to call for displaying the results
        virtual void registerDisplayCallback(boost::function<void(cv::Mat)>& f);
        virtual void registerDisplayCallback(boost::function<void(cv::cuda::GpuMat)>& f);
        virtual void spawnDisplay();
        virtual void killDisplay();

        /******************* Children *******************************/
        virtual int  addChild(boost::shared_ptr<Node> child);
        virtual void removeChild(boost::shared_ptr<Node> child);
        virtual void removeChild(int idx);

        template<typename T> boost::shared_ptr< TypedParameter<T> > getParameter(std::string name)
        {
            for(int i = 0; i < parameters.size(); ++i)
            {
                if(parameters[i]->name == name)
                    return boost::dynamic_pointer_cast<T, Node>(parameters[i]);
            }
            return boost::shared_ptr<T>();
        }
        virtual bool subParameterExists(std::string name)
        {
            for(int i = 0; i < childParameters.size(); ++i)
            {
                if(childParameters[i].second->name == name)
                {
                    return true;
                }
            }
            return false;
        }
        template<typename T> bool checkSubParameterType(std::string name)
        {
            for(int i = 0; i < childParameters.size(); ++i)
            {
                if(childParameters[i].second->name == name)
                {
                    return boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(childParameters[i]) != NULL;
                }
            }
        }
        template<typename T> boost::shared_ptr< TypedParameter<T> > getSubParameter(std::string name)
        {
            for(int i = 0; i < childParameters.size(); ++i)
            {
                if(childParameters[i].second->name == name)
                {
                    return boost::dynamic_pointer_cast<TypedParameter<T>, Parameter>(childParameters[i].second);
                }
            }
            return boost::shared_ptr< TypedParameter<T> >(); // Return a NULL pointer
        }


        boost::function<void(std::string)>      errorCallback;
        boost::function<void(std::string)>      warningCallback;
        std::vector< boost::shared_ptr<Node> >  children;
        boost::shared_ptr<Node> parent;
        const std::string nodeName;     // Constant name that describes the node ie: Sobel
        std::string treeName;           // Name as placed in the tree ie: Sobel1
        // Parameters of this node
        std::vector< boost::shared_ptr< Parameter > > parameters;
        // Parameters of the child, paired with the index of the child
        std::vector< std::pair< int, boost::shared_ptr< Parameter > > > childParameters;

        boost::function<void(cv::Mat)>          cpuCallback;
        boost::function<void(cv::cuda::GpuMat)> gpuCallback;
        bool drawResults;


    };
}





