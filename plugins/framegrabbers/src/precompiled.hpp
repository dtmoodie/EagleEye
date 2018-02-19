#pragma once
#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#include <Dbt.h>
#include <Wmcodecdsp.h>
#include <assert.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfplay.h>
#include <mfreadwrite.h>
#include <new>
#include <shlwapi.h>
#include <limits>
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
#ifdef HAVE_GSTREAMER
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#endif

