#include "NodeView.h"

NodeView::NodeView(QWidget* parent) :
    QGraphicsView(parent), currentWidget(nullptr), resizeGrabSize(20), rightClickMenu(new QMenu(this))
{}

NodeView::NodeView(QGraphicsScene *scene, QWidget *parent):
    QGraphicsView(scene, parent), currentWidget(nullptr), resizeGrabSize(20), rightClickMenu(new QMenu(this))
{
    actions.push_back(new QAction("Delete Node", rightClickMenu));
    rightClickMenu->addActions(actions);

    connect(rightClickMenu, SIGNAL(triggered(QAction*)), this, SLOT(on_actionSelect(QAction*)));
}
void NodeView::addWidget(QGraphicsProxyWidget * widget, ObjectId id)
{
    widgetMap[id] = widget;
    // If this widget has a parent, draw a line to it
    drawLine2Parent(widget);
}
QGraphicsProxyWidget* NodeView::getWidget(ObjectId id)
{
    auto itr = widgetMap.find(id);
    if (itr != widgetMap.end())
        return itr->second;
    else return nullptr;
}
void NodeView::on_actionSelect(QAction *action)
{
    if(action == actions[0])
    {
        // Delete the current node
        auto nodeWidget = dynamic_cast<QNodeWidget*>(currentWidget->widget());
        if(nodeWidget)
        {
            auto node = nodeWidget->getNode();

            auto parent = node->getParent();
            if(parent)
                parent->removeChild(node->GetObjectId());
            // Remove this node from the node manager
            // This causes a deadlock due to this (gui) thread blocking, while waiting for stopThread to finish
            // But some UI calls to imshow operate on the main thread.... Maybe those should be handled via a
            // queued connection to the main thread.
            emit stopThread();
            emit selectionChanged(nullptr);
            emit widgetDeleted(nodeWidget);
            // Causes segfault sometimes, processing thread has fully exited but segfault still occurs in ~QNodeWidget
            // Looks like it has to do with the QGraphicsProxyWidget trying to handle deletion.  Maybe we need to delete that first?
            // Or maybe we just need to delete the QNodeWidget first.
            scene()->removeItem(currentWidget);
            delete currentWidget;
            currentWidget = nullptr;
            EagleLib::NodeManager::getInstance().removeNode(node->GetObjectId());
            emit startThread();
        }
    }
}

void NodeView::mousePressEvent(QMouseEvent* event)
{
    if (QGraphicsItem* item = itemAt(event->pos().x(), event->pos().y()))
    {
        if (QGraphicsProxyWidget* widget = dynamic_cast<QGraphicsProxyWidget*>(item))
        {
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
                if(event->button() == Qt::RightButton)
                {

                    // Spawn the right click dialog
                    QPoint pos = mapToGlobal(mousePressPosition);
                    rightClickMenu->popup(pos);
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
    QGraphicsView::mouseReleaseEvent(event);
    resize = false;
}
QGraphicsLineItem* NodeView::drawLine2Parent(QGraphicsProxyWidget* child)
{
    QNodeWidget* nodeWidget = dynamic_cast<QNodeWidget*>(child->widget());
    if(nodeWidget == nullptr)
        return nullptr;
    EagleLib::Node* node = nodeWidget->getNode();
    if(!node)
        return nullptr;
    auto parentId = node->parentId;
    if(!parentId.IsValid())
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
        if(QGraphicsProxyWidget* parentWidget = getWidget(parentId))
        {
            auto center = parentWidget->pos() += QPointF(parentWidget->size().width()/2, parentWidget->size().height()/2);
            // Draw line from this widget to parent
            connectingLine->setLine(center.x(), center.y(), child->pos().x() + child->size().width()/2, child->pos().y() + child->size().height()/2);
        }
    }
    return connectingLine;
}
