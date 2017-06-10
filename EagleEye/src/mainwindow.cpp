#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/core/IDataStream.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <nodes/FlowScene>
#include <nodes/FlowView>
#include <nodes/DataModelRegistry>
#include <RuntimeObjectSystem/ObjectInterface.h>
#include <IObject.h>
namespace aq{
    class NodeOutProxy: virtual public QtNodes::NodeData{
    public:
        NodeOutProxy(const rcc::shared_ptr<Nodes::Node>& node_):
            node(node_){}
        virtual QtNodes::NodeDataType type() const{
            QtNodes::NodeDataType out;
            out.name = QString::fromStdString(node->getTreeName());
            out.id = QString::fromStdString("aq::Nodes::Node");
            return out;
        }
        rcc::shared_ptr<Nodes::Node> node;
    };

    class DSOutProxy: virtual public QtNodes::NodeData{
    public:
        DSOutProxy(const rcc::shared_ptr<IDataStream>& ds): m_ds(ds){
        }

        virtual QtNodes::NodeDataType type() const {
            QtNodes::NodeDataType out;
            out.name = QString::fromStdString("ds");
            out.id = QString::fromStdString("aq::Nodes::Node");
            return out;
        }
            
        rcc::shared_ptr<IDataStream> m_ds;
    };


    class ParamOutProxy: virtual public NodeOutProxy{
    public:
        ParamOutProxy(mo::IParam* param_, const rcc::shared_ptr<Nodes::Node>& node_):
        param(param_), NodeOutProxy(node_){
        }
        virtual QtNodes::NodeDataType type() const{
            QtNodes::NodeDataType out;
            out.name = QString::fromStdString(param->getName());
            out.id = QString::fromStdString(param->getTypeInfo().name());
            return out;
        }
        mo::IParam* param;
    };

    class NodeProxy: virtual public QtNodes::NodeDataModel{
    public:
        NodeProxy(rcc::shared_ptr<aq::Nodes::Node>&& obj):
            m_obj(std::move(obj)){}
        virtual QString caption() const{
            return QString::fromStdString(m_obj->getTreeName());
        }
        virtual QString name() const{return caption();}
        virtual std::unique_ptr<QtNodes::NodeDataModel> clone()const{
            IObjectConstructor* ctr = m_obj->GetConstructor();
            rcc::shared_ptr<aq::Nodes::Node> node(ctr->Construct());
            node->Init(true);
            return std::unique_ptr<QtNodes::NodeDataModel>(new NodeProxy(std::move(node)));
        }

        virtual unsigned int nPorts(QtNodes::PortType portType) const{
            switch(portType){
                case QtNodes::PortType::In: return m_obj->getInputs().size() + 1;
                case QtNodes::PortType::Out: return m_obj->getOutputs().size() + 1;
                case QtNodes::PortType::None: return 0;
            }
            return 0;
        }

        virtual QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const{
            QtNodes::NodeDataType output;
            switch(portType){
                case QtNodes::PortType::In: {
                    if(portIndex == 0){
                        output.id = "aq::Nodes::Node";
                        output.name = "parents";
                    }else{
                        auto inputs = m_obj->getInputs();
                        if(portIndex <= inputs.size() && portIndex > 0){
                            output.id = QString::fromStdString(inputs[portIndex - 1]->getTypeInfo().name());
                            output.name = QString::fromStdString(inputs[portIndex - 1]->getName());
                        }
                    }
                    break;
                }
                case QtNodes::PortType::Out: {
                    if(portIndex == 0){
                        output.id = "aq::Nodes::Node";
                        output.name = "children";
                    }else{
                        auto outputs = m_obj->getOutputs();
                        if(portIndex > 0 && portIndex <= outputs.size()){
                            output.id = QString::fromStdString(outputs[portIndex - 1]->getTypeInfo().name());
                            output.name = QString::fromStdString(outputs[portIndex - 1]->getName());
                        }
                    }
                }
            case QtNodes::PortType::None:
                break;
            }
            return output;
        }

        virtual void setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port){
            if(port == 0){
                std::shared_ptr<NodeOutProxy> typed = std::dynamic_pointer_cast<NodeOutProxy>(nodeData);
                if(typed){
                    m_obj->addParent(typed->node.get());
                    return;
                }
                std::shared_ptr<DSOutProxy> ds = std::dynamic_pointer_cast<DSOutProxy>(nodeData);
                if(ds){
                    ds->m_ds->addNode(m_obj);
                }
            }
            std::shared_ptr<ParamOutProxy> typed = std::dynamic_pointer_cast<ParamOutProxy>(nodeData);
            if(typed){
                auto inputs = m_obj->getInputs();
                if(port >= 0 && port < inputs.size()){
                    m_obj->connectInput(typed->node, typed->param, inputs[port]);
                }
            }
        }

        virtual std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex port){
            if(port == 0){
                return std::make_shared<NodeOutProxy>(m_obj);
            }
            auto outputs = m_obj->getOutputs();
            if(port > 0 && port <= outputs.size()){
                return std::make_shared<ParamOutProxy>(outputs[port - 1], m_obj);
            }
            return std::shared_ptr<QtNodes::NodeData>();
        }
        virtual QWidget * embeddedWidget(){
            // TODO
            return nullptr;
        }

        rcc::shared_ptr<aq::Nodes::Node> m_obj;
    };
    class DataStreamProxy : virtual public QtNodes::NodeDataModel {
    public:
        DataStreamProxy(rcc::shared_ptr<aq::IDataStream>&& ds = rcc::shared_ptr<aq::IDataStream>()) :
            m_obj(std::move(ds)) {
        }

        virtual QString caption() const { return "DataStream"; }
        virtual QString name() const { return "DataStream"; }
        virtual std::unique_ptr<QtNodes::NodeDataModel> clone() const {
            return std::unique_ptr<DataStreamProxy>(new DataStreamProxy(std::move(aq::IDataStream::create("", ""))));
        }
        virtual unsigned int nPorts(QtNodes::PortType portType) const {
            if (portType == QtNodes::PortType::Out) return 1;
            return 0;
        }
        virtual QtNodes::NodeDataType dataType(QtNodes::PortType port_type, QtNodes::PortIndex port_index) const {
            QtNodes::NodeDataType output;
            if (port_type == QtNodes::PortType::Out) {
                output.id = "aq::Nodes::Node";
                output.name = "children";
            }
            return output;
        }
        virtual void setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port) {

        }
        virtual std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex port) {
            return std::make_shared<DSOutProxy>(m_obj);
        }
        virtual QWidget * embeddedWidget() {
            // TODO
            return nullptr;
        }

        rcc::shared_ptr<aq::IDataStream> m_obj;
    };

    class DataStreamConstructor: virtual public QtNodes::NodeDataModel{
    public:
        DataStreamConstructor(){}
        virtual QString caption() const { return "DataStream"; }
        virtual QString name() const { return "DataStream"; }
        virtual unsigned int nPorts(QtNodes::PortType portType) const {
            if (portType == QtNodes::PortType::Out) return 1;
            return 0;
        }
        virtual std::unique_ptr<QtNodes::NodeDataModel> clone() const {
            return std::unique_ptr<DataStreamProxy>(new DataStreamProxy(std::move(aq::IDataStream::create("", ""))));
        }
        virtual QtNodes::NodeDataType dataType(QtNodes::PortType port_type, QtNodes::PortIndex port_index) const {
            QtNodes::NodeDataType output;
            if (port_type == QtNodes::PortType::Out) {
                output.id = "aq::Nodes::Node";
                output.name = "children";
            }
            return output;
        }
        virtual void setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port) {

        }
        virtual std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex port) {
            return std::shared_ptr<QtNodes::NodeData>();
        }
        virtual QWidget * embeddedWidget() {
            // TODO
            return nullptr;
        }
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
    auto ctrs = mo::MetaObjectFactory::instance()->getConstructors(aq::Nodes::Node::s_interfaceID);
    std::vector<int> ids;
    std::vector<std::string> interfces_names;
    for(auto ctr : ctrs){
        ids.push_back(ctr->GetInterfaceId());
        interfces_names.push_back(ctr->GetInterfaceName());
    }
    registry->registerModel<aq::DataStreamConstructor>("DataStream");
    ctrs = mo::MetaObjectFactory::instance()->getConstructors();
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
