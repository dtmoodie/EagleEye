#include "openni2_initializer.h"
#include "OpenNI.h"
#include <opencv2/core.hpp>
#include <ObjectInterfacePerModule.h>
#include <Aquila/rcc/SystemTable.hpp>


using namespace aq;
initializer_NI2::initializer_NI2()
{
    openni::Status rc = openni::OpenNI::initialize();
    CV_Assert(rc == openni::STATUS_OK);
}

initializer_NI2* initializer_NI2::instance()
{
    static initializer_NI2* inst = nullptr;
    if(inst == nullptr)
    {
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        if(table)
        {
            inst = table->GetSingleton<initializer_NI2>();
            if(inst == nullptr)
            {
                inst = new initializer_NI2();
                table->SetSingleton<initializer_NI2>(inst);
            }
        }else
        {
            inst = new initializer_NI2();
        }
    }
    return inst;
}
