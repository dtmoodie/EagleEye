
#include "EagleLib/utilities/CudaCallbacks.hpp"
#include "UI/InterThread.hpp"
#include "plotters/Plotter.h"
#include "qcustomplot.h"
#include "remotery.h"
#include <EagleLib/Project_defs.hpp>
#include "RuntimeLinkLibrary.h"
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_core300d.lib")
RUNTIME_COMPILER_LINKLIBRARY("libParameterd.lib")
RUNTIME_COMPILER_LINKLIBRARY("QCustomPlotd.lib")
RUNTIME_COMPILER_LINKLIBRARY("EagleLibd.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5OpenGLd.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5PrintSupportd.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgetsd.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5Guid.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5Cored.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("QCustomPlot.lib")
RUNTIME_COMPILER_LINKLIBRARY("EagleLib.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5OpenGL.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5PrintSupport.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5Widgets.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5Gui.lib")
RUNTIME_COMPILER_LINKLIBRARY("Qt5Core.lib")
#endif
SETUP_PROJECT_DEF