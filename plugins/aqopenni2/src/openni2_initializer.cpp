#include "openni2_initializer.h"
#include "OpenNI.h"
#include <MetaObject/core/SystemTable.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <opencv2/core.hpp>

using namespace aq;
initializer_NI2::initializer_NI2()
{
    openni::Status rc = openni::OpenNI::initialize();
    CV_Assert(rc == openni::STATUS_OK);
}

initializer_NI2* initializer_NI2::instance()
{
    return singleton<initializer_NI2>();
}
