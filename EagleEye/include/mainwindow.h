#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/signals/detail/SignalMacros.hpp>
#include "GraphScene.hpp"
namespace aq
{
    class IDataStream;
}

namespace Ui {
    class MainWindow;
}

class GraphScene;
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
    void setVmSm(std::map<std::string, std::string>* vm, std::map<std::string, std::string>* sm){
        this->sm = sm; this->vm = vm;
        _graph_scene->setVmSm(vm, sm);
    }
signals:
    void sigCallbackAdded();
private slots:
    void on_action_load_triggered();

    void on_action_save_triggered();
    void handleGuiCallback();
protected:
    // called from non gui thread
    void onGuiPush();
    
    GraphScene* _graph_scene;
    Ui::MainWindow* ui;
    std::map<std::string, std::string>* sm = nullptr;
    std::map<std::string, std::string>* vm = nullptr;
};

#endif // MAINWINDOW_H
