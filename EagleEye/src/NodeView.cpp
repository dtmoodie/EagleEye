#include "NodeView.h"
#include "qapplication.h"
#include "qdrag.h"
#include <qmimedata.h>
NodeView::NodeView(QWidget* parent) :
    QGraphicsView(parent), currentWidget(nullptr), resizeGrabSize(20), rightClickMenu(new QMenu(this))
{}

NodeView::NodeView(QGraphicsScene *scene, QWidget *parent):
    QGraphicsView(scene, parent), currentWidget(nullptr), resizeGrabSize(20), rightClickMenu(new QMenu(this))
{
    actions.push_back(new QAction("Delete Node", rightClickMenu));
    actions.push_back(new QAction("Display as image", rightClickMenu));
    actions.push_back(new QAction("Plot", rightClickMenu));
    actions[1]->setEnabled(false);
    actions[2]->setEnabled(false);
    rightClickMenu->addActions(actions);
    connect(actions[0], SIGNAL(triggered()), this, SLOT(on_deleteNode()));
    connect(actions[1], SIGNAL(triggered()), this, SLOT(on_displayImage()));
    connect(actions[2], SIGNAL(triggered()), this, SLOT(on_plotData()));

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
void NodeView::on_parameter_clicked(EagleLib::Parameter::Ptr param)
{
    currentParam = param;
}

void NodeView::on_deleteNode()
{
    // Delete the current node
    if(currentWidget == nullptr)
        return;
    auto nodeWidget = dynamic_cast<QNodeWidget*>(currentWidget->widget());
    if(nodeWidget)
    {
        auto node = nodeWidget->getNode();
        node->enabled = false;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(30));
        emit stopThread();
        auto parent = node->getParent();
        if(parent != nullptr)
            parent->removeChild(node);
        emit selectionChanged(nullptr);
        emit widgetDeleted(nodeWidget);
        scene()->removeItem(currentWidget);
        delete currentWidget;
        currentWidget = nullptr;
        emit startThread();
    }
    return;
}
void NodeView::on_displayImage()
{
    if(currentParam != nullptr)
        emit displayImage(currentParam);
}
void NodeView::on_plotData()
{
    if(currentParam != nullptr)
        emit plotData(currentParam);
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
                        fileName = EagleLib::NodeManager::getInstance().getNodeFile(id);
                }
                QDrag* drag = new QDrag(this);
                QMimeData* mimeData = new QMimeData();
                QList<QUrl> urls;

                urls << QUrl::fromLocalFile(QString::fromStdString(fileName));
                mimeData->setUrls(urls);
                drag->setMimeData(mimeData);
                Qt::DropAction dropAction = drag->exec();
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
                    QGraphicsView::mousePressEvent(event);
                    // Spawn the right click dialog
                    if(currentParam!= nullptr)
                    {
                        if(currentParam->typeInfo == Loki::TypeInfo(typeid(cv::cuda::GpuMat))           ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(cv::cuda::GpuMat*))         ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(cv::cuda::GpuMat&))         ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(cv::Mat))                   ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(cv::Mat*))                  ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(cv::Mat&))                  ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<int>))          ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<float>))        ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<double>))       ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<cv::Vec2b>))    ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<cv::Vec2f>))    ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<cv::Vec2d>))    ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<cv::Vec3b>))    ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<cv::Vec3f>))    ||
                            currentParam->typeInfo == Loki::TypeInfo(typeid(std::vector<cv::Vec3d>)))
                        {
                            actions[1]->setEnabled(true);
                            actions[2]->setEnabled(true);
                            actions[1]->setText("Display as image (" + QString::fromStdString(currentParam->name) + ")");
                            actions[2]->setText("Plot (" + QString::fromStdString(currentParam->name) + ")");
                        }
                    }else
                    {
                        actions[1]->setEnabled(false);
                        actions[2]->setEnabled(false);
                    }
                    QPoint pos = mapToGlobal(mousePressPosition);
                    rightClickMenu->popup(pos);
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
    QGraphicsView::mouseReleaseEvent(event);
    resize = false;
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

QGraphicsLineItem* NodeView::drawLine2Parent(QGraphicsProxyWidget* child)
{
    QNodeWidget* nodeWidget = dynamic_cast<QNodeWidget*>(child->widget());
    if(nodeWidget == nullptr)
        return nullptr;
    EagleLib::Node::Ptr node = nodeWidget->getNode();
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
        auto parentPtr = node->getParent();
        if(parentPtr != nullptr)
        {
            QGraphicsProxyWidget* parentWidget = getWidget(parentPtr->GetObjectId());
            if(parentWidget)
            {
                auto center = parentWidget->pos() += QPointF(parentWidget->size().width()/2, parentWidget->size().height()/2);
                // Draw line from this widget to parent
                connectingLine->setLine(center.x(), center.y(), child->pos().x() + child->size().width()/2, child->pos().y() + child->size().height()/2);
            }
        }
    }
    return connectingLine;
}
QGraphicsProxyWidget* NodeView::getParent(EagleLib::Node::Ptr child)
{
    auto parentPtr = child->getParent();
    if(parentPtr != nullptr)
    {
        QGraphicsProxyWidget* parentWidget = getWidget(parentPtr->GetObjectId());
        return parentWidget;
    }
    return nullptr;
}
