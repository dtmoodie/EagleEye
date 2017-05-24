#include "NodeView.h"
#include "qapplication.h"
#include "qdrag.h"
#include <qmimedata.h>

#include <Aquila/nodes/NodeFactory.hpp>
#include "signal_dialog.h"
#include <Aquila/plotters/PlotManager.h>

NodeView::NodeView(QWidget* parent) :
    QGraphicsView(parent), currentWidget(nullptr), resizeGrabSize(20), rightClickMenu(new QMenu(this))
{}

NodeView::NodeView(QGraphicsScene *scene, QWidget *parent):
    QGraphicsView(scene, parent), currentWidget(nullptr), resizeGrabSize(20), rightClickMenu(new QMenu(this))
{
    actions.push_back(new QAction("Delete Node / stream", rightClickMenu));
    actions.push_back(new QAction("Display Signals", rightClickMenu));
    actions.push_back(new QAction("Display as image", rightClickMenu));
    actions.push_back(new QAction("Plot", rightClickMenu));
    
    actions[2]->setEnabled(false);
    actions[3]->setEnabled(false);
    rightClickMenu->addActions(actions);
    rightClickMenu->installEventFilter(this);
    actions[0]->installEventFilter(this);
    actions[1]->installEventFilter(this);
    actions[2]->installEventFilter(this);
    actions[3]->installEventFilter(this);
    connect(actions[0], SIGNAL(triggered()), this, SLOT(on_deleteNode()));
    connect(actions[1], SIGNAL(triggered()), this, SLOT(on_display_signals()));
    connect(actions[2], SIGNAL(triggered()), this, SLOT(on_displayImage()));
    connect(actions[3], SIGNAL(triggered()), this, SLOT(on_plotData()));
    currentParam = nullptr;
    _signal_dialog = nullptr;
}
void NodeView::addWidget(QGraphicsProxyWidget * widget, ObjectId id)
{
    widgetMap[id] = widget;
    // If this widget has a parent, draw a line to it
    drawLine2Parent(widget);
}
void NodeView::addWidget(QGraphicsProxyWidget* widget, aq::IDataStream* stream_)
{
    dataStreamWidget[stream_] = widget;
}
QGraphicsProxyWidget* NodeView::getWidget(ObjectId id)
{
    auto itr = widgetMap.find(id);
    if (itr != widgetMap.end())
        return itr->second;
    else return nullptr;
}
void NodeView::removeWidget(ObjectId id)
{
    widgetMap.erase(id);
}
QGraphicsProxyWidget* NodeView::getWidget(aq::IDataStream* id)
{
    auto itr = dataStreamWidget.find(id);
    if (itr != dataStreamWidget.end())
        return itr->second;
    else return nullptr;
}
void NodeView::on_parameter_clicked(mo::IParam* param, QPoint pos)
{
    // Spawn the right click dialog
    if (param != nullptr)
    {
        if (aq::PlotManager::Instance()->CanPlotParameter(param))
        {
            actions[2]->setEnabled(true);
            actions[3]->setEnabled(true);
            actions[2]->setText("Display as image (" + QString::fromStdString(param->GetName()) + ")");
            actions[3]->setText("Plot (" + QString::fromStdString(param->GetName()) + ")");
            currentParam = param;
        }
    }
    else
    {
        actions[2]->setEnabled(false);
        actions[3]->setEnabled(false);
    }
    //rightClickMenu->popup(mapToGlobal(pos));
}

void NodeView::on_deleteNode()
{
    // Delete the current node
    if(currentWidget == nullptr)
        return;
    auto itr = parentLineMap.find(currentWidget);
    if (itr != parentLineMap.end())
        delete itr->second;
    if(auto nodeWidget = dynamic_cast<QNodeWidget*>(currentWidget->widget()))
    {
        
        if (nodeWidget)
        {
            auto node = nodeWidget->getNode();
            boost::this_thread::sleep_for(boost::chrono::milliseconds(30));
            emit stopThread();
            boost::this_thread::sleep_for(boost::chrono::milliseconds(30));

            //Parameters::UI::ProcessingThreadCallbackService::run();
            auto parents = node->GetParents();
            for(auto parent : parents)
                parent->RemoveChild(node);

            emit selectionChanged(nullptr);
            emit widgetDeleted(nodeWidget);
            emit startThread();
        }
    }
    if(auto streamWidget = dynamic_cast<DataStreamWidget*>(currentWidget->widget()))
    {
        auto stream = streamWidget->GetStream();
        stream->StartThread();
        emit widgetDeleted(streamWidget);
        dataStreamWidget.erase(stream.get());
    }
    scene()->removeItem(currentWidget);
    delete currentWidget;
    currentWidget = nullptr;
    
    return;
}
void NodeView::on_displayImage()
{
    currentWidget = nullptr;
    if(currentParam != nullptr)
        emit displayImage(currentParam);
}
void NodeView::on_plotData()
{
    currentWidget = nullptr;
    if(currentParam != nullptr)
        emit plotData(currentParam);
}
void NodeView::on_display_signals()
{
    if(currentWidget)
    {
        auto widget = currentWidget->widget();
        if(auto node_widget = dynamic_cast<QNodeWidget*>(widget))
        {
            auto node = node_widget->getNode();
            if(_signal_dialog)
            {
                _signal_dialog->update(node->GetDataStream()->GetRelayManager());
            }else
            {
                _signal_dialog = new signal_dialog(node->GetDataStream()->GetRelayManager(), this);
            }
            _signal_dialog->show();
        }
        if(auto stream_widget = dynamic_cast<DataStreamWidget*>(widget))
        {
            auto stream = stream_widget->GetStream();
            if(_signal_dialog)
            {
                _signal_dialog->update(stream->GetRelayManager());
            }else
            {
                _signal_dialog = new signal_dialog(stream->GetRelayManager(), this);
            }
            _signal_dialog->show();
        }
    }
}

void NodeView::mousePressEvent(QMouseEvent* event)
{
    if (QGraphicsItem* item = itemAt(event->pos().x(), event->pos().y()))
    {
        if (QGraphicsProxyWidget* widget = dynamic_cast<QGraphicsProxyWidget*>(item))
        {
            if(event->button() == Qt::MiddleButton)
            {
                QNodeWidget* nodeWidget = dynamic_cast<QNodeWidget*>(widget->widget());
                auto node = nodeWidget->getNode();
                std::string fileName;
                if(node != nullptr)
                {
                    auto id = node->GetObjectId();
                    if(id.IsValid())
                        fileName = aq::NodeFactory::Instance()->GetNodeFile(id);
                }
                QDrag* drag = new QDrag(this);
                QMimeData* mimeData = new QMimeData();
                QList<QUrl> urls;
                urls << QUrl::fromLocalFile(QString::fromStdString(fileName));
                mimeData->setUrls(urls);
                mimeData->setData("application/x-qt-windows-mime;value=\"FileName\"",QByteArray(fileName.c_str()));

                drag->setMimeData(mimeData);
                drag->exec();
                return QGraphicsView::mousePressEvent(event);
            }
            mousePressPosition = event->pos();
            currentWidget = widget;
            emit selectionChanged(widget);
            if(event->button() == Qt::LeftButton)
            {
                // If within the 5x5 corners of the widget, resize
                auto pos = widget->pos();
                auto size = widget->size();
                grabPoint = mapToScene(event->pos());
                int x = grabPoint.x();
                int y = grabPoint.y();
                if(x < pos.x() + resizeGrabSize && y < pos.y() + resizeGrabSize)// Top left corner
                {
                    corner = 0;
                    resize = true;
                }
                if(x < pos.x() + resizeGrabSize && y > pos.y() + size.height() - resizeGrabSize)    // bottom left corner
                {
                    corner = 3;
                    resize = true;
                }
                if(x > pos.x() + size.width() - 5 && y < pos.y() + resizeGrabSize)   // Top right
                {
                    corner = 1;
                    resize = true;
                }
                if(x > pos.x() + size.width() - 5 && y > pos.y() + size.height() - resizeGrabSize) // Bottom right
                {
                    corner = 2;
                    resize = true;
                }
            }else
            {
                if(event->button() == Qt::RightButton)
                {
                    actions[2]->setEnabled(false);
                    actions[3]->setEnabled(false);    
                    auto pos = event->pos();
                    rightClickMenu->popup(mapToGlobal(pos));
                    QGraphicsView::mousePressEvent(event);

                }
            }
        }
    }else
    {
        QGraphicsView::mousePressEvent(event);
    }
}
void NodeView::mouseMoveEvent(QMouseEvent* event)
{
    if (currentWidget)
    {
        if(resize)
        {
            QPointF point = mapToScene(event->pos());
            QPointF resizeVector = grabPoint - point;
            if(corner == 0)
            {
                // Resize the widget by the amount and move to match mouse
                currentWidget->resize( currentWidget->size().width() + resizeVector.x(), currentWidget->size().height() + resizeVector.y());
                currentWidget->setPos(currentWidget->pos() - resizeVector);
            }
            if(corner == 1)
            {
                currentWidget->resize( currentWidget->size().width() - resizeVector.x(), currentWidget->size().height() + resizeVector.y());
            }
            if(corner == 2)
            {
                currentWidget->resize( currentWidget->size().width() - resizeVector.x(), currentWidget->size().height() - resizeVector.y());
            }
            if(corner == 3)
            {
                currentWidget->resize( currentWidget->size().width() - resizeVector.x(), currentWidget->size().height() - resizeVector.y());
            }
            grabPoint = point;
            return QGraphicsView::mouseMoveEvent(event);
        }
        QPointF pos = currentWidget->pos();
        pos += event->pos() - mousePressPosition;
        currentWidget->setPos(pos);
        mousePressPosition = event->pos();
        // Check if the node has a parrent
        drawLine2Parent(currentWidget);
    }
    QGraphicsView::mouseMoveEvent(event);
}
void NodeView::mouseReleaseEvent(QMouseEvent* event)
{
    currentWidget = nullptr;
    resize = false;
    QGraphicsView::mouseReleaseEvent(event);

}
void NodeView::wheelEvent(QWheelEvent* event)
{
    if(QApplication::keyboardModifiers() & Qt::ControlModifier)
    {
        setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        static double scaleFactor = 1.15;
        if(event->delta() > 0)
        {
            scale(scaleFactor, scaleFactor);
        }else
        {
            scale(1.0/scaleFactor, 1.0 / scaleFactor);
        }
    }

    return QGraphicsView::wheelEvent(event);
}
bool NodeView::eventFilter(QObject *object, QEvent *event)
{
    if(object == rightClickMenu)
    {
        if(event->type() == QEvent::MouseButtonRelease)
        {
            //currentWidget = nullptr;
            resize = false;
            return false;
        }
    }
    if(event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseButtonDblClick || event->type() == QEvent::MouseButtonRelease)
    {
        QMouseEvent* mev = dynamic_cast<QMouseEvent*>(event);
        if(mev)
        {
            if(mev->button() == Qt::RightButton)
            {
                return true; // Ignore right clicks in this menu
            }
        }
    }
    return false;
}

QGraphicsLineItem* NodeView::drawLine2Parent(QGraphicsProxyWidget* child)
{
    QNodeWidget* nodeWidget = dynamic_cast<QNodeWidget*>(child->widget());
    if(nodeWidget == nullptr)
        return nullptr;
    auto node = nodeWidget->getNode();
    if(node == nullptr)
        return nullptr;
    // First check if this child's line already exists
    auto itr = parentLineMap.find(child);
    QGraphicsLineItem* connectingLine = nullptr;
    if(itr != parentLineMap.end())
    {
        // Child already exists, update
        connectingLine = itr->second;
    }else
    {
        connectingLine = new QGraphicsLineItem();
        scene()->addItem(connectingLine);
        parentLineMap[child] = connectingLine;
        connectingLine->setZValue(-1);
    }
    if(nodeWidget)
    {
        auto parents = node->GetParents();
        for(auto parent : parents)
        {
            QGraphicsProxyWidget* parentWidget = getWidget(parent->GetObjectId());
            if (parentWidget)
            {
                auto center = parentWidget->pos() += QPointF(parentWidget->size().width() / 2, parentWidget->size().height() / 2);
                // Draw line from this widget to parent
                connectingLine->setLine(center.x(), center.y(), child->pos().x() + child->size().width() / 2, child->pos().y() + child->size().height() / 2);
            }
        }
        
        /*{
            auto streamWidget = getWidget(node->GetDataStream());
            if(streamWidget)
            {
                auto center = streamWidget->pos() += QPointF(streamWidget->size().width() / 2, streamWidget->size().height() / 2);
                connectingLine->setLine(center.x(), center.y(), child->pos().x() + child->size().width() / 2, child->pos().y() + child->size().height() / 2);
            } 
        }*/
    }
    return connectingLine;
}
QGraphicsProxyWidget* NodeView::getParent(aq::Nodes::Node::Ptr child)
{
    auto parents = child->GetParents();
    for(auto parent : parents)
    {
        // TODO figure out how this should correctly be used
        QGraphicsProxyWidget* parentWidget = getWidget(parent->GetObjectId());
        return parentWidget;
    }
    return nullptr;
}
std::vector<QGraphicsProxyWidget*> NodeView::getParents(aq::Nodes::Node::Ptr child)
{
    std::vector<QGraphicsProxyWidget*> output;
    auto parents = child->GetParents();
    for (auto parent : parents)
    {
        // TODO figure out how this should correctly be used
        QGraphicsProxyWidget* parentWidget = getWidget(parent->GetObjectId());
        output.push_back(parentWidget);
    }
    return output;
}
QGraphicsProxyWidget* NodeView::getStream(aq::IDataStream* stream_id)
{
    auto itr = dataStreamWidget.find(stream_id);
    if(itr != dataStreamWidget.end())
    {
        return itr->second;
    }
    return nullptr;
}
