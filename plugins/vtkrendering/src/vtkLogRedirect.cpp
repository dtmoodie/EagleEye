#include "vtkLogRedirect.h"
#include "EagleLib/rcc/SystemTable.hpp"
#include "ObjectInterfacePerModule.h"
#include <MetaObject/Logging/Log.hpp>

void vtkLogRedirect::init()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto instance = table->GetSingleton<vtkLogRedirect>();
        if (instance == nullptr)
        {
            instance = new vtkLogRedirect();
            table->SetSingleton(instance);
            vtkOutputWindow::SetInstance(instance);
        }
    }
}
void vtkLogRedirect::DisplayText(const char* msg)
{
    LOG(debug) << msg;
    LOG(debug) << mo::print_callstack(0, true);
}
void vtkLogRedirect::DisplayErrorText(const char* msg)
{
    LOG(error) << msg;
    LOG(debug) << mo::print_callstack(0, true);
}
void vtkLogRedirect::DisplayWarningText(const char* msg)
{
    LOG(warning) << msg;
    LOG(debug) << mo::print_callstack(0, true);
}
void vtkLogRedirect::DisplayGenericWarningText(const char* msg)
{
    LOG(warning) << msg;
    LOG(debug) << mo::print_callstack(0, true);
}