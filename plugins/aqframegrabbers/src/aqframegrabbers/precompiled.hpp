#pragma once
#ifdef _MSC_VER
#define NOMINMAX
#include <Dbt.h>
#include <Wmcodecdsp.h>
#include <assert.h>
#include <limits>
#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfplay.h>
#include <mfreadwrite.h>
#include <new>
#include <shlwapi.h>
#include <windows.h>
#pragma comment(lib, "Mfplat.lib")
#pragma comment(lib, "Mf.lib")
#endif

#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include <Aquila/framegrabbers/FrameGrabberInfo.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
