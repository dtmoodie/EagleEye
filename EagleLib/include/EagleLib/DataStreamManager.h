#pragma once
#include "EagleLib/Detail/Export.hpp"
#include "EagleLib/rcc/SystemTable.hpp"
#include <MetaObject/IMetaObject.hpp>

#include <shared_ptr.hpp>
#include <type_traits>
#include <vector>
#include <memory>
namespace mo
{
    RelayManager;
}


namespace EagleLib
{
    namespace Nodes
    {
        class Node;
    }
    class IViewManager;
    class ICoordinateManager;
    class IRenderEngine;
    class ITrackManager;
    class IFrameGrabber;
    class IParameterBuffer;
    class IVariableSink;

    class EAGLE_EXPORTS IDataStream: public TInterface<IID_DataStream, mo::IMetaObject>
    {
    public:


        typedef rcc::shared_ptr<IDataStream> Ptr;
        static Ptr create(const std::string& document, const std::string& preferred_frame_grabber = "");
        static bool CanLoadDocument(const std::string& document);

        virtual ~IDataStream();

        // Handles user interactions such as moving the viewport, user interface callbacks, etc
        virtual rcc::weak_ptr<IViewManager>            GetViewManager() = 0;

        // Handles conversion of coordinate systems, such as to and from image coordinates, world coordinates, render scene coordinates, etc.
        virtual rcc::weak_ptr<ICoordinateManager>      GetCoordinateManager() = 0;
        
        // Handles actual rendering of data.  Use for adding extra objects to the scene
        virtual rcc::weak_ptr<IRenderEngine>           GetRenderingEngine() = 0;
        
        // Handles tracking objects within a stream and communicating with the global track manager to track across multiple data streams
        virtual rcc::weak_ptr<ITrackManager>            GetTrackManager() = 0;

        // Handles actual loading of the image, etc
        virtual rcc::weak_ptr<IFrameGrabber>           GetFrameGrabber() = 0;

        virtual mo::RelayManager*                            GetRelayManager() = 0;
        virtual std::shared_ptr<mo::IVariableManager> GetVariableManager() = 0;

        virtual IParameterBuffer*                        GetParameterBuffer() = 0;

        virtual std::vector<rcc::shared_ptr<Nodes::Node>> GetNodes() = 0;

        virtual bool LoadDocument(const std::string& document, const std::string& prefered_loader = "") = 0;

        virtual std::vector<rcc::shared_ptr<Nodes::Node>> AddNode(const std::string& nodeName) = 0;
        virtual void AddNode(rcc::shared_ptr<Nodes::Node> node) = 0;
        virtual void AddNodes(std::vector<rcc::shared_ptr<Nodes::Node>> node) = 0;
        virtual void RemoveNode(rcc::shared_ptr<Nodes::Node> node) = 0;
        virtual void RemoveNode(Nodes::Node* node) = 0;
        
        virtual void StartThread() = 0;
        virtual void StopThread() = 0;
        virtual void PauseThread() = 0;
        virtual void ResumeThread() = 0;
        virtual void process() = 0;
        
        virtual void AddVariableSink(IVariableSink* sink) = 0;
        virtual void RemoveVariableSink(IVariableSink* sink) = 0;
        
        // Get or create singleton for this datastream
        template<typename T> T* GetSingleton()
        {
            auto& ptr_ref = GetSingleton(mo::TypeInfo(typeid(T)));
            if(ptr_ref)
            {
                return static_cast<Singleton<T>*>(ptr_ref.get())->ptr;
            }else
            {
                ptr_ref.reset(new Singleton<T>(new T()));
            }
        }

        template<typename T> rcc::weak_ptr<T> GetIObjectSingleton()
        {
            auto& ptr_ref = GetIObjectSingleton(mo::TypeInfo(typeid(T)));
            if(ptr_ref)
            {
                return static_cast<IObjectSingleton<T>*>(ptr_ref.get())->ptr;
            }else
            {
                auto ptr = new IObjectSingleton<T>(T::create());
                ptr_ref.reset(ptr);
                return ptr->ptr;
            }
        }

        // Transfers ownership to the datastream
        template<typename T> T* SetSingleton(typename std::enable_if<!std::is_base_of<IObject, T>::value, T>* singleton)
        {
            auto& ptr_ref = GetSingleton(mo::TypeInfo(typeid(T)));
            ptr_ref.reset(new Singleton<T>(singleton));
            return singleton;
        }
        template<typename T> rcc::weak_ptr<T> SetSingleton(typename std::enable_if<std::is_base_of<IObject, T>::value, T>* singleton)
        {
            auto& ptr_ref = GetIObjectSingleton(mo::TypeInfo(typeid(T)));
            auto sig_ptr = new IObjectSingleton<T>(singleton);
            ptr_ref.reset(sig_ptr);
            return sig_ptr->ptr;;
        }
    protected:
        virtual std::unique_ptr<ISingleton>& GetSingleton(mo::TypeInfo type) = 0;
        virtual std::unique_ptr<ISingleton>& GetIObjectSingleton(mo::TypeInfo type) = 0;
    };
}
