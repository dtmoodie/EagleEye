#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <Aquila/Nodes/Node.h>
#include <Aquila/Nodes/IFrameGrabber.hpp>
#include <MetaObject/MetaObjectFactory.hpp>
#include <nodes/FlowScene>
#include <nodes/FlowView>
#include <nodes/DataModelRegistry>
#include <RuntimeObjectSystem/ObjectInterface.h>
#include <IObject.h>
namespace aq{
    class ParamOutProxy: virtual public QtNodes::NodeData{
    public:
        ParamOutProxy(mo::IParameter* param_, const rcc::shared_ptr<Nodes::Node>& node_):
        param(param_), node(node_){
        }
        virtual QtNodes::NodeDataType type() const{
            QtNodes::NodeDataType out;
            out.name = QString::fromStdString(param->GetName());
            out.id = QString::fromStdString(param->GetTypeInfo().name());
            return out;
        }
        mo::IParameter* param;
        rcc::shared_ptr<Nodes::Node> node;
    };

    class NodeProxy: virtual public QtNodes::NodeDataModel{
    public:
        NodeProxy(rcc::shared_ptr<aq::Nodes::Node>&& obj):
            m_obj(std::move(obj)){}
        virtual QString caption() const{
            return QString::fromStdString(m_obj->GetTreeName());
        }
        virtual QString name() const{return caption();}
        virtual std::unique_ptr<QtNodes::NodeDataModel> clone()const{
            IObjectConstructor* ctr = m_obj->GetConstructor();
            rcc::shared_ptr<aq::Nodes::Node> node(ctr->Construct());
            node->Init(true);
            return std::unique_ptr<QtNodes::NodeDataModel>(new NodeProxy(std::move(node)));
        }

        virtual
        unsigned int nPorts(QtNodes::PortType portType) const{
            switch(portType)
            {
                case QtNodes::PortType::In: return m_obj->GetInputs().size();
                case QtNodes::PortType::Out: return m_obj->GetOutputs().size();
            }

            return 0;
        }

        virtual
        QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const{
            QtNodes::NodeDataType output;
            switch(portType)
            {
                case QtNodes::PortType::In: {
                    auto inputs = m_obj->GetInputs();
                    if(portIndex < inputs.size() && portIndex >=0){
                        output.id = QString::fromStdString(inputs[portIndex]->GetTypeInfo().name());
                        output.name = QString::fromStdString(inputs[portIndex]->GetName());
                    }
                    break;
                }
                case QtNodes::PortType::Out: {
                    auto outputs = m_obj->GetOutputs();
                    if(portIndex >= 0 && portIndex < outputs.size()){
                        output.id = QString::fromStdString(outputs[portIndex]->GetTypeInfo().name());
                        output.name = QString::fromStdString(outputs[portIndex]->GetName());
                    }
                }
            }
            return output;
        }

        virtual
        void
        setInData(std::shared_ptr<QtNodes::NodeData> nodeData,
                  QtNodes::PortIndex port){
            // TODO
            std::shared_ptr<ParamOutProxy> typed = std::dynamic_pointer_cast<ParamOutProxy>(nodeData);
            if(typed){
                auto inputs = m_obj->GetInputs();
                if(port >= 0 && port < inputs.size()){
                    m_obj->ConnectInput(typed->node, typed->param, inputs[port]);
                }
            }
        }

        virtual
        std::shared_ptr<QtNodes::NodeData>
        outData(QtNodes::PortIndex port){
            auto outputs = m_obj->GetOutputs();
            if(port>= 0 && port < outputs.size()){
                return std::make_shared<ParamOutProxy>(outputs[port], m_obj);
            }
            return std::shared_ptr<QtNodes::NodeData>();
        }
        virtual
        QWidget *
        embeddedWidget(){
            // TODO
            return nullptr;
        }

        rcc::shared_ptr<aq::Nodes::Node> m_obj;
    };

    class NodeConstructor: virtual public QtNodes::NodeDataModel{
    public:
        NodeConstructor(IObjectConstructor* constructor): m_constructor(constructor){}
        virtual QString caption() const{return m_constructor->GetName();}
        virtual QString name() const{return caption();}
        virtual std::unique_ptr<QtNodes::NodeDataModel> clone() const{
            rcc::shared_ptr<aq::Nodes::Node> node(m_constructor->Construct());
            node->Init(true);
            return std::unique_ptr<QtNodes::NodeDataModel>(new NodeProxy(std::move(node)));
        }

        virtual
        unsigned int nPorts(QtNodes::PortType portType) const{
            return 0;
        }

        virtual
        QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const{
            return QtNodes::NodeDataType();
        }

        virtual
        void
        setInData(std::shared_ptr<QtNodes::NodeData> nodeData,
                  QtNodes::PortIndex port){
            // TODO
        }

        virtual
        std::shared_ptr<QtNodes::NodeData>
        outData(QtNodes::PortIndex port){
            return std::shared_ptr<QtNodes::NodeData>();
        }
        virtual
        QWidget *
        embeddedWidget(){
            // TODO
            return nullptr;
        }
        IObjectConstructor* m_constructor;
    };
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow){
    ui->setupUi(this);
    std::shared_ptr<QtNodes::DataModelRegistry> registry = std::make_shared<QtNodes::DataModelRegistry>();
    auto ctrs = mo::MetaObjectFactory::Instance()->GetConstructors(aq::Nodes::Node::s_interfaceID);
    std::vector<int> ids;
    std::vector<std::string> interfces_names;
    for(auto ctr : ctrs){
        ids.push_back(ctr->GetInterfaceId());
        interfces_names.push_back(ctr->GetInterfaceName());
    }

    ctrs = mo::MetaObjectFactory::Instance()->GetConstructors();
    for(auto ctr : ctrs){
        if(ctr->GetInterfaceName() == aq::Nodes::Node::GetInterfaceName())
            registry->registerModel<aq::NodeConstructor>("nodes", std::make_unique<aq::NodeConstructor>(ctr));
    }
    for(auto ctr : ctrs){
        if(ctr->GetInterfaceName() == aq::Nodes::IFrameGrabber::GetInterfaceName())
            registry->registerModel<aq::NodeConstructor>("Frame grabbers", std::make_unique<aq::NodeConstructor>(ctr));
    }

    _flow_scene = new QtNodes::FlowScene(registry);
    ui->main_layout->addWidget(new QtNodes::FlowView(_flow_scene));

}

MainWindow::~MainWindow(){
    delete ui;
}

MO_REGISTER_CLASS(MainWindow)
