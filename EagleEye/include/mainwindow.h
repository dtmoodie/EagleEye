#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/signals/detail/SignalMacros.hpp>

namespace aq
{
    class IDataStream;
}

namespace Ui {
    class MainWindow;
}
namespace QtNodes{
    class FlowScene;
}

class MainWindow : public QMainWindow, public mo::IMetaObject
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    MO_BEGIN(MainWindow)
        MO_SIGNAL(void, StartThreads);
        MO_SIGNAL(void, StopThreads);
        MO_SIGNAL(void, PauseThreads);
        MO_SIGNAL(void, ResumeThreads);
    MO_END;
protected:
    QtNodes::FlowScene* _flow_scene;
    Ui::MainWindow* ui;
};

#endif // MAINWINDOW_H
