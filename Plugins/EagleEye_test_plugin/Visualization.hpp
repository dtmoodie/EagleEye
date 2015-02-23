#pragma once
#include <QObject>
#include "../../EagleLib/include/Factory.h"
#include "../../EagleLib/include/nodes/Node.h"
#include <list>


class VisualizationFactory: public QObject, public NodePluginFactory
{
    Q_OBJECT
    Q_INTERFACES(NodePluginFactory)

    Q_PLUGIN_METADATA(IID "myTestPlugin")
public:
    VisualizationFactory(QObject* parent = 0);
    virtual ~VisualizationFactory() {}
    virtual QStringList nodes();
    virtual EagleLib::Node::Ptr createNode(const std::string &nodeName, QWidget* parent);
private:
    std::map<std::string, EagleLib::NodeFactory::Ptr> factories;
};



