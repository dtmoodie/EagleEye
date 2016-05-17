#include <qapplication.h>
#include <qwidget.h>
#include <qmainwindow.h>

#include <PythonQt/PythonQt.h>
#include <PythonQt/PythonQt_QtAll.h>
#include <PythonQt/gui/PythonQtScriptingConsole.h>

#include "PythonInterface.h"

int main(int argc, char** argv)
{
	QApplication qapp(argc, argv);
	PythonQt::init(PythonQt::IgnoreSiteModule | PythonQt::RedirectStdOut);
	PythonQt_QtAll::init();
	PythonQtObjectPtr  mainContext = PythonQt::self()->getMainModule();
	PythonQtScriptingConsole console(NULL, mainContext);
	EagleLib::python::wrappers::RegisterMetaTypes();
	EagleLib::python::EaglePython main;
	
	mainContext.addObject("EaglePython", &main);
	console.show();
	return qapp.exec();
}