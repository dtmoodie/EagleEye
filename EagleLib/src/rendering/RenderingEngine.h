#pragma once
#include <memory>
#include "IObject.h"

namespace EagleLib
{
	// Render objects are objects that are rendered inside of a scene
	class IRenderObject : TInterface<IID_RenderObject, IObject>
	{
	public:
		virtual void Render() = 0;
	};

	// Render scene holds all objects in a scene, enables / disables specific objects, etc.
	class IRenderScene : TInterface<IID_RenderScene, IObject>
	{
	public:
		virtual void Render() = 0;
		virtual void AddRenderObject(std::shared_ptr<IRenderObject> obj) = 0;
	};


	// Render engine handles actually calling render, update, etc
	class IRenderEngine: TInterface<IID_RenderEngine, IObject>
	{
	public:
		virtual void Render() = 0;
		virtual void AddRenderObject(std::shared_ptr<IRenderObject> obj) = 0;
		virtual void AddRenderScene(std::shared_ptr<IRenderScene> scene) = 0;
	};
}