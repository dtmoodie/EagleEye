#pragma once

#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/framegrabbers/FrameGrabberInfo.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#ifdef HAVE_GSTREAMER
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>
#endif

#ifdef _MSC_VER
#include <new>
#include <windows.h>
#include <mfobjects.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <Wmcodecdsp.h>
#include <assert.h>
#include <Dbt.h>
#include <shlwapi.h>
#include <mfplay.h>
#pragma comment(lib, "Mfplat.lib")
#pragma comment(lib, "Mf.lib")
#else

#endif