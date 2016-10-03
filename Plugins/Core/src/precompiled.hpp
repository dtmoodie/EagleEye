#pragma once

#include <EagleLib/Nodes/Node.h>
#include <EagleLib/Nodes/NodeInfo.hpp>
#include "EagleLib/Detail/PluginExport.hpp"
#include "EagleLib/DataStreamManager.h"
#include <EagleLib/ObjectDetection.hpp>
#include <EagleLib/utilities/CudaUtils.hpp>
#include "EagleLib/utilities/CudaCallbacks.hpp"
#include "EagleLib/rcc/SystemTable.hpp"



#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include <IRuntimeObjectSystem.h>



#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include <EagleLib/rcc/external_includes/cv_imgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_highgui.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_imgcodec.hpp>
#include <EagleLib/rcc/external_includes/cv_cudabgsegm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudafeatures2d.hpp>
#include <EagleLib/rcc/external_includes/cv_cudafilters.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudalegacy.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaobjdetect.hpp>
#include <opencv2/core/opengl.hpp>

#include <boost/lexical_cast.hpp>

#include <algorithm>
#include <utility>