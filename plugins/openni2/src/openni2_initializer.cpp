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
    static std::shared_ptr<initializer_NI2> inst = nullptr;
    if (inst == nullptr)
    {
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        if (table)
        {
            inst = table->getSingleton<initializer_NI2>();
            if (inst == nullptr)
            {
                inst = std::make_shared<initializer_NI2>();
                table->setSingleton<initializer_NI2>(inst);
            }
        }
        else
        {
            inst = std::make_shared<initializer_NI2>();
        }
    }
    return inst.get();
}
