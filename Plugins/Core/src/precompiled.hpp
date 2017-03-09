#pragma once

#include <Aquila/Nodes/Node.h>
#include <Aquila/Nodes/NodeInfo.hpp>
#include "Aquila/Detail/PluginExport.hpp"
#include "Aquila/IDataStream.hpp"
#include <Aquila/ObjectDetection.hpp>
#include <Aquila/utilities/CudaUtils.hpp>
#include "Aquila/utilities/CudaCallbacks.hpp"
#include "Aquila/rcc/SystemTable.hpp"



#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include <IRuntimeObjectSystem.h>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"


#include <Aquila/rcc/external_includes/cv_core.hpp>
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>
#include <Aquila/rcc/external_includes/cv_highgui.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>
#include <Aquila/rcc/external_includes/cv_cudabgsegm.hpp>
#include <Aquila/rcc/external_includes/cv_cudafeatures2d.hpp>
#include <Aquila/rcc/external_includes/cv_cudafilters.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudalegacy.hpp>
#include <Aquila/rcc/external_includes/cv_cudaobjdetect.hpp>
#include <Aquila/rcc/external_includes/cv_cudaoptflow.hpp>
#include <opencv2/core/opengl.hpp>

#include <boost/lexical_cast.hpp>

#include <algorithm>
#include <utility>