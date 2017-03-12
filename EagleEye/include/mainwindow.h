#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#ifndef NODE_EDITOR_SHARED
#define NODE_EDITOR_SHARED
#endif

#include <NodeEditor/NodeData.hpp>
#include <NodeEditor/FlowScene.hpp>
#include <NodeEditor/FlowView.hpp>
#include <NodeEditor/ConnectionStyle.hpp>

#include <QMainWindow>

namespace aq
{
    class IDataStream;
}

namespace Ui {
class MainWindow;
}
class SettingDialog;
class bookmark_dialog;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    std::shared_ptr<QtNodes::DataModelRegistry> _node_registry;
    QtNodes::FlowScene* scene;
    Ui::MainWindow *                                    ui;
};

#endif // MAINWINDOW_H
