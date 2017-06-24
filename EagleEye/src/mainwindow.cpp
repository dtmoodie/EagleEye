#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <MetaObject/MetaParameters.hpp>
#include <nodes/FlowScene>
#include <nodes/FlowView>
#include <nodes/DataModelRegistry>
#include <RuntimeObjectSystem/ObjectInterface.h>
#include <IObject.h>
#include "Aquila/nodes/Node.hpp"
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include <qwidget.h>
#include <qmenubar.h>

#include "GraphScene.hpp"
#include "object_proxies.hpp"
#include "object_constructors.hpp"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow){
    mo::MetaParams::initialize();
    ui->setupUi(this);
    std::shared_ptr<QtNodes::DataModelRegistry> registry = std::make_shared<QtNodes::DataModelRegistry>();
    auto ctrs = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::Node::s_interfaceID);
    std::vector<int> ids;
    std::vector<std::string> interfces_names;
    for(auto ctr : ctrs){
        ids.push_back(ctr->GetInterfaceId());
        interfces_names.push_back(ctr->GetInterfaceName());
    }
    registry->registerModel<aq::DataStreamConstructor>("DataStream");
    ctrs = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::Node::s_interfaceID);
    for(auto ctr : ctrs){
        registry->registerModel<aq::NodeConstructor>("nodes", std::make_unique<aq::NodeConstructor>(ctr));
    }
    ctrs = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::IFrameGrabber::s_interfaceID);
    for(auto ctr : ctrs){
        registry->registerModel<aq::FrameGrabberConstructor>("Frame grabbers", std::make_unique<aq::FrameGrabberConstructor>(ctr));
    }

    _graph_scene  = new GraphScene(registry);
    ui->main_layout->addWidget(new QtNodes::FlowView(_graph_scene));
}

MainWindow::~MainWindow(){
    delete ui;
}

MO_REGISTER_CLASS(MainWindow)

void MainWindow::on_action_load_triggered()
{
    _graph_scene->load();
}

void MainWindow::on_action_save_triggered()
{
    _graph_scene->save(true);
}
