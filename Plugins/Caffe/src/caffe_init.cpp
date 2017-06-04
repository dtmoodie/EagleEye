#include "caffe_init.h"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "Aquila/rcc/SystemTable.hpp"

#include <caffe/caffe.hpp>
using namespace aq;

caffe_init_singleton::caffe_init_singleton()
{
    int count = 1;
    char** argv = new char*[1]; // "./EagleEye" };
    argv[0] = ".EagleEye";
    ::caffe::GlobalInit(&count, &argv);
    ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}


caffe_init_singleton* caffe_init_singleton::inst()
{
    static caffe_init_singleton* g_inst = nullptr;
    if (g_inst == nullptr)
    {
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        if (!(g_inst = table->getSingleton<caffe_init_singleton>()))
        {
            g_inst = new caffe_init_singleton();
            table->setSingleton<caffe_init_singleton>(g_inst);
        }
    }
    return g_inst;
}
