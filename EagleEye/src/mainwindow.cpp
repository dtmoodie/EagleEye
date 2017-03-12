#define PARAMETERS_GENERATE_UI
#define HAVE_OPENCV
#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "FileOrFolderDialog.h"
#include <qfiledialog.h>
#include <qgraphicsproxywidget.h>
#include "QGLWidget"
#include <QGraphicsSceneMouseEvent>

#include "settingdialog.h"
#include "dialog_network_stream_selection.h"
#include <QNodeWidget.h>

#include "bookmark_dialog.h"
//#include <GL/gl.h>
//#include <GL/glu.h>


#include <shared_ptr.hpp>
#include <Aquila/rcc/SystemTable.hpp>
#include <Aquila/utilities/ogl_allocators.h>
#include "Aquila/utilities/CpuMatAllocators.h"
#include <Aquila/Logging.h>

#include "Aquila/utilities/BufferPool.hpp"

#include "Aquila/Signals.h"
#include "Aquila/logger.hpp"
#include "Aquila/Plugins.h"
#include <Aquila/Nodes/Node.h>
#include <Aquila/Nodes/NodeFactory.h>
#include <Aquila/utilities/ColorMapperFactory.hpp>

#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Parameters/UI/WidgetFactory.hpp>

#include <signal.h>
using namespace QtNodes;
struct NodeProxy;
struct ParameterProxy: public NodeData
{
    ParameterProxy(mo::IParameter* param, rcc::shared_ptr<aq::Nodes::Node> node, NodeProxy* proxy, Node* display_node):
        _param(param),
        _node(node),
        _node_proxy(proxy),
        _display_node(display_node)
    {}

    bool sameType(NodeData const &nodeData) const
    {
        //return dynamic_cast<const ParameterProxy&>(nodeData)._param->GetTypeInfo() == _param->GetTypeInfo();
        return nodeData.type().id == QString::fromStdString(_param->GetTypeInfo().name());
    }

    /// Type for inner use
    NodeDataType type() const
    {
        return {QString::fromStdString(_param->GetTypeInfo().name()),
                QString::fromStdString(_param->GetTreeName())};
    }
    mo::IParameter* _param;
    rcc::shared_ptr<aq::Nodes::Node> _node;
    NodeProxy* _node_proxy;
    Node* _display_node;
};

struct ParentProxy: public NodeData
{

    ParentProxy(rcc::shared_ptr<aq::Nodes::Node> node):
        _node(node)
    {

    }

    bool sameType(NodeData const &nodeData) const
    {
        return nodeData.type().id == QString::fromStdString(mo::TypeInfo(typeid(aq::Nodes::Node)).name());
    }

    /// Type for inner use
    NodeDataType type() const
    {
        return {QString::fromStdString(mo::TypeInfo(typeid(aq::Nodes::Node)).name()),
                QString::fromStdString(_node->GetTreeName())};
    }
    rcc::shared_ptr<aq::Nodes::Node> _node;
};
struct DataStreamData: public NodeData
{
    DataStreamData(rcc::shared_ptr<aq::IDataStream> ds):
        _ds(ds)
    {
    }

    NodeDataType type() const
    {
        return {"DataStream", QString::number(_ds->GetPerTypeId())};
    }
    rcc::shared_ptr<aq::IDataStream> _ds;
};
struct NodeProxy: public QtNodes::NodeDataModel
{
    NodeProxy(rcc::shared_ptr<aq::Nodes::Node> node, FlowScene* scene):
        _node(node),
        _scene(scene)
    {

    }

    QString name() const
    {
        return QString::fromStdString(_node->GetTreeName());
    }

    QString caption() const
    {
        return QString::fromStdString(_node->GetTreeName());
    }

    bool resizable() const { return true; }

    unsigned int nPorts(QtNodes::PortType portType) const
    {
        if(portType == PortType::In)
        {
            auto inputs = _node->GetInputs();
            return inputs.size() + 1;
        }else
        {
            auto outputs = _node->GetOutputs();
            return outputs.size() + 1;
        }
    }

    QtNodes::NodeDataType dataType(QtNodes::PortType portType, PortIndex portIndex) const
    {
        if(portType == PortType::In)
        {
            if(portIndex == 0)
            {
                return {QString::fromStdString(mo::TypeInfo(typeid(aq::Nodes::Node)).name()),
                        "Parents"};
            }
            else
            {
                auto inputs = _node->GetInputs();
                return {QString::fromStdString(inputs[portIndex - 1]->GetTypeInfo().name()),
                            QString::fromStdString(inputs[portIndex - 1]->GetTreeName())};
            }
        }else
        {
            if(portIndex == 0)
            {
                return {QString::fromStdString(mo::TypeInfo(typeid(aq::Nodes::Node)).name()),
                        "Children"};
            }
            else
            {
            auto outputs = _node->GetOutputs();
            return {QString::fromStdString(outputs[portIndex - 1]->GetTypeInfo().name()),
                    QString::fromStdString(outputs[portIndex - 1]->GetTreeName())};
            }
        }
    }

    void setInData(std::shared_ptr<NodeData> nodeData,
              PortIndex port)
    {
        if(port == 0)
        {
            if(nodeData->type().id == "DataStream")
            {
                auto typed = std::dynamic_pointer_cast<DataStreamData>(nodeData);
                if(typed)
                {
                    _node->SetDataStream(typed->_ds.Get());
                }
            }else
            {
                auto typed = std::dynamic_pointer_cast<ParentProxy>(nodeData);
                if(typed)
                {
                    _node->AddParent(typed->_node.Get());
                }
            }
        }else
        {
            std::vector<mo::InputParameter*> inputs = _node->GetInputs();
            auto typed = std::dynamic_pointer_cast<ParameterProxy>(nodeData);
            if(typed)
            {
                if(_node->ConnectInput(typed->_node, typed->_param, inputs[port - 1]))
                {
                    //_connections.push_back(_scene->createConnection(*this->_display_node, 0, *(typed->_display_node), 0));
                    _scene->createConnection(*this->_display_node, 0, *(typed->_display_node), 0);
                }
            }
        }
    }

    std::shared_ptr<NodeData> outData(PortIndex port)
    {
        if(port == 0)
        {
            return std::shared_ptr<NodeData>(new ParentProxy(_node));
        }else
        {
            std::vector<mo::IParameter*> outputs = _node->GetOutputs();
            return std::shared_ptr<NodeData>(new ParameterProxy(outputs[port - 1], _node, this, _display_node));
        }
    }

    QWidget * embeddedWidget()
    {
        std::vector<mo::IParameter*> params = _node->GetAllParameters();
        int control_count = 0;
        for(auto param : params)
        {
            if(param->CheckFlags(mo::Control_e))
            {
                ++control_count;
            }
        }

        if(control_count)
        {
            if(!_container)
            {
                _container = new QWidget();
                QVBoxLayout* layout = new QVBoxLayout();
                _container->setLayout(layout);
                for(auto param : params)
                {
                    if(param->CheckFlags(mo::Control_e))
                    {
                        auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(param);
                        if(proxy)
                        {
                            _proxies.push_back(proxy);
                            auto widget = proxy->GetParameterWidget(_container);
                            if(widget)
                            {
                                layout->addWidget(widget);
                            }
                        }
                    }
                }
            }
            return _container;
        }
        return nullptr;
    }

    std::unique_ptr<NodeDataModel> clone() const
    {
        IObjectConstructor* constructor = _node->GetConstructor();
        IObject* obj = constructor->Construct();
        obj->Init(true);
        return std::unique_ptr<NodeDataModel>(new NodeProxy(obj, _scene));
    }

    void setNode(Node* node)
    {
        _display_node = node;
    }

    rcc::shared_ptr<aq::Nodes::Node> _node;
    std::vector<std::shared_ptr<mo::UI::qt::IParameterProxy>> _proxies;
    FlowScene* _scene;
    //std::vector<std::shared_ptr<QtNodes::Connection>> _connections;
    Node* _display_node;
    QWidget* _container = nullptr;
};

struct NodeConstructorProxy: public QtNodes::DataModelConstructor
{
    NodeConstructorProxy(IObjectConstructor* ctor, FlowScene* scene):
        constructor(ctor),
        _scene(scene)
    {

    }

    QString name() const
    {
        return constructor->GetName();
    }
    std::unique_ptr<QtNodes::NodeDataModel> construct() const
    {
        IObject* obj = constructor->Construct();
        obj->Init(true);
        return std::unique_ptr<NodeDataModel>(new NodeProxy(obj, _scene));
    }

    IObjectConstructor* constructor;
    FlowScene* _scene;
};



struct DataStreamProxy: public QtNodes::NodeDataModel
{
    DataStreamProxy(rcc::shared_ptr<aq::IDataStream> ds):
        _ds(ds)
    {

    }
    QString caption() const
    {
        return "DataStream";
    }
    QString name() const
    {
        return "DataStream";
    }
    QWidget * embeddedWidget()
    {
        return nullptr;
    }
    NodeDataType dataType(QtNodes::PortType portType, PortIndex portIndex) const
    {
        return {"DataStream", QString::number(_ds->GetPerTypeId())};
    }
    std::shared_ptr<NodeData> outData(PortIndex port)
    {
        return std::shared_ptr<NodeData>(new DataStreamData(_ds));
    }
    void setInData(std::shared_ptr<NodeData> nodeData,
              PortIndex port)
    {

    }
    unsigned int nPorts(QtNodes::PortType portType) const
    {
        if(portType == PortType::In)
        {
            return 0;
        }else
        {
            return 1;
        }
    }
    std::unique_ptr<NodeDataModel> clone() const
    {
        return {};
    }
    rcc::shared_ptr<aq::IDataStream> _ds;
};

struct DatastreamConstructor: public QtNodes::DataModelConstructor
{
    QString name() const
    {
        return "DataStream";
    }
    std::unique_ptr<QtNodes::NodeDataModel> construct() const
    {
        auto ds = aq::IDataStream::Create();
        if(ds)
        {
            return std::unique_ptr<NodeDataModel>(new DataStreamProxy(ds));
        }
        return {};
    }
};

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    _node_registry.reset(new QtNodes::DataModelRegistry());
    scene = new QtNodes::FlowScene(_node_registry);
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(aq::Nodes::Node::s_interfaceID);
    for(auto constructor : constructors)
    {
        _node_registry->registerModel(std::make_unique<NodeConstructorProxy>(constructor, scene));
    }


    ui->gridLayout->removeWidget(ui->console);
    ui->gridLayout->addWidget(new FlowView(scene), 1, 0, 1, 4);
    ui->gridLayout->addWidget(ui->console, 2, 0, 1, 4);
}

MainWindow::~MainWindow()
{

}

