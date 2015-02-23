#include "nodes/Root.h"
#include <boost/chrono.hpp>
#include <boost/date_time.hpp>
/*
using namespace EagleLib;

#ifdef RCC_ENABLED
#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"


#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cuda")
#endif

void Root::Init(bool isFirstInit)
{

}

#endif

Root::Root()
{
#ifdef RCC_ENABLED
    m_pRuntimeObjSystem.reset(new RuntimeObjectSystem());

    if(!m_pRuntimeObjSystem->Initialise(nullptr, nullptr))
    {
        m_pRuntimeObjSystem.reset();
    }else
    {
//        m_pRuntimeObjSystem->GetObjectFactorySystem()->AddListener(this);
    }


#endif
    nodeName = "Root";
}

Root::~Root()
{

}

cv::cuda::GpuMat
Root::doProcess(cv::cuda::GpuMat& img)
{
#ifdef RCC_ENABLED
    static boost::posix_time::ptime prevTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = boost::posix_time::microsec_clock::universal_time() - prevTime;
  //  objSystem->GetFileChangeNotifier()->Update(delta.total_milliseconds());

    if(objSystem->GetIsCompiling() && statusCallback)
            statusCallback(std::string("Recompiling"));
    if(objSystem->GetIsCompiledComplete())
        objSystem->LoadCompiledModule();
#endif
    for(auto itr = children.begin(); itr != children.end(); ++itr)
    {
        img = (*itr)->process(img);
    }
    return img;
}

REGISTERCLASS(Root)

*/
