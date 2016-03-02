//
// Copyright (c) 2010-2011 Matthew Jack and Doug Binks
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

////////////////////////////////////////////////////////////////////////////////////////////////////////
// 
// IObject header file.
//
// The RuntimeCompiler library does not declare an IObject interface, only forward declares it.
// Hence each project can define their own base interface for objects they want to runtime compile
// and construct by using their own declaration of IObject in their own header file.
//
////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#ifndef IOBJECT_INCLUDED
#define IOBJECT_INCLUDED

#include "ObjectInterface.h"
#include "ISimpleSerializer.h"
#include <iostream>
#include <algorithm>
#include <assert.h>

struct ISimpleSerializer;
class ObjectFactorySystem;
template<typename T> class shared_ptr;

// IIDs
enum InterfaceIDEnum
{
	IID_IOBJECT,
    IID_NodeObject,
    IID_Plotter,
	IID_SignalHandler,
	IID_RenderEngine,
	IID_RenderScene,
	IID_RenderObject,
	IID_RenderObjectConstructor,
	IID_RenderInteractor,
	IID_RenderObjectFactory,
    IID_Algorithm,
	IID_CudaAlgorithm,
    IID_ViewManager,
    IID_CoordinateManager,
    IID_TrackManager,
    IID_FrameGrabber,
	IID_ENDInterfaceID
};

typedef unsigned int InterfaceID;

// Template to help with IIDs
template< InterfaceID Tiid, typename TSuper> struct TInterface : public TSuper
{
	static const InterfaceID s_interfaceID = Tiid;
    virtual IObject* GetInterface( InterfaceID _iid)
    {
        switch(_iid)
        {
        case Tiid:
            return this;
            break;
        default:
            return TSuper::GetInterface(_iid);
        }
    }
};




// IObject itself below is a special case as the base class
// Also it doesn't hurt to have it coded up explicitly for reference

struct IObject
{
    static const InterfaceID s_interfaceID = IID_IOBJECT;

    virtual IObject* GetInterface(InterfaceID __iid)
    {
        switch(__iid)
        {
        case IID_IOBJECT:
            return this;
        default:
            return nullptr;
        }
    }

    template< typename T> void GetInterface( T** pReturn )
    {
        GetInterface( T::s_interfaceID, (void**)pReturn );
    }


    IObject() : _isRuntimeDelete(false) {}
    virtual ~IObject()
    {
		
    }

    // Perform any object initialization
    // Should be called with isFirstInit=true on object creation
    // Will automatically be called with isFirstInit=false whenever a system serialization is performed
    virtual void Init( bool isFirstInit )
    {

    }

    //return the PerTypeObjectId of this object, which is unique per class
    virtual PerTypeObjectId GetPerTypeId() const = 0;

    virtual void GetObjectId( ObjectId& id ) const
    {
        id.m_ConstructorId = GetConstructor()->GetConstructorId();
        id.m_PerTypeId = GetPerTypeId();
    }
    virtual ObjectId GetObjectId() const
    {
        ObjectId ret;
        GetObjectId( ret );
        return ret;
    }


    //return the constructor for this class
    virtual IObjectConstructor* GetConstructor() const = 0;

    //serialise is not pure virtual as many objects do not need state
    virtual void Serialize(ISimpleSerializer *pSerializer)
    {
        

    }
	void SerializeNotifiers(ISimpleSerializer* pSerializer)
	{
		
	}
    virtual const char* GetTypeName() const = 0;

    
    virtual void updateParent(){}

protected:
    bool IsRuntimeDelete() { return _isRuntimeDelete; }

private:
    friend class ObjectFactorySystem;

    // Set to true when object is being deleted because a new version has been created
    // Destructor should use this information to not delete other IObjects in this case
    // since these objects will still be needed
    bool _isRuntimeDelete;
};

#endif //IOBJECT_INCLUDED
