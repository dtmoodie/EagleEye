#pragma once
#include <MetaObject/IMetaObject.hpp>
#include <memory>

namespace mo
{
    class IParameter;
}

namespace EagleLib
{
    // Render objects are objects that are rendered inside of a scene
    class IRenderObject : public TInterface<IID_RenderObject, mo::IMetaObject>
    {
    public:
        virtual void Render() = 0;
    };

    class IRenderObjectConstructor : public TInterface<IID_RenderObjectConstructor, mo::IMetaObject>
    {
    public:
        virtual std::shared_ptr<IRenderObject> Construct(std::shared_ptr<mo::IParameter> param) = 0;
    };

    class IRenderObjectFactory : public TInterface<IID_RenderObject, mo::IMetaObject>
    {
    public:
        virtual std::shared_ptr<IRenderObject> Create(std::shared_ptr<mo::IParameter> param) = 0;
        static void RegisterConstructorStatic(std::shared_ptr<IRenderObjectConstructor> constructor);

        virtual void RegisterConstructor(std::shared_ptr<IRenderObjectConstructor> constructor) = 0;
    };

    // Render scene holds all objects in a scene, enables / disables specific objects, etc.
    class IRenderScene : public TInterface<IID_RenderScene, IObject>
    {
    public:
        virtual void Render() = 0;
        virtual void AddRenderObject(std::shared_ptr<IRenderObject> obj) = 0;
    };

    class IRenderInteractor : TInterface<IID_RenderInteractor, IObject>
    {
    public:

    };

    // Render engine handles actually calling render, update, etc
    class IRenderEngine: public TInterface<IID_RenderEngine, IObject>
    {
    public:
        virtual void Render() = 0;
        virtual void AddRenderScene(std::shared_ptr<IRenderScene> scene) = 0;
    };
}