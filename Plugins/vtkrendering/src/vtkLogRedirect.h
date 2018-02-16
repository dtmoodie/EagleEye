#pragma once
#include "vtkOutputWindow.h"

class PLUGIN_EXPORTS vtkLogRedirect: public vtkOutputWindow
{
public:
    static void init();
    virtual void DisplayText(const char*);
    virtual void DisplayErrorText(const char*);
    virtual void DisplayWarningText(const char*);
    virtual void DisplayGenericWarningText(const char*);
};