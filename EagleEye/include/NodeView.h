#pragma once
#include <qgraphicsview.h>
#include <qevent.h>
#include <qgraphicsproxywidget.h>
#include <EagleLib/nodes/Node.h>
#include <QNodeWidget.h>
#include <QMenu>
#include <EagleLib/rcc/shared_ptr.hpp>
class NodeView : public QGraphicsView
{
    Q_OBJECT
public:

    NodeView(QWidget* parent = 0);

    NodeView(QGraphicsScene *scene, QWidget *parent = 0);
    void addWidget(QGraphicsProxyWidget * widget, ObjectId id);
    void addWidget(QGraphicsProxyWidget* widget, EagleLib::IDataStream* stream_);
    QGraphicsProxyWidget* getWidget(ObjectId id);
    QGraphicsProxyWidget* getWidget(EagleLib::IDataStream* stream_);
    
    void removeWidget(ObjectId id);
    void mousePressEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);
    void wheelEvent(QWheelEvent* event);
    QGraphicsLineItem* drawLine2Parent(QGraphicsProxyWidget* child);
    QGraphicsProxyWidget* getParent(EagleLib::Nodes::Node::Ptr child);
    QGraphicsProxyWidget* getStream(EagleLib::IDataStream* stream_id);

signals:
    void selectionChanged(QGraphicsProxyWidget* widget);
    void stopThread();
    void startThread();
    void widgetDeleted(QNodeWidget*);
    void widgetDeleted(DataStreamWidget*);
    void plotData(Parameters::Parameter* param);
    void displayImage(Parameters::Parameter* param);
private slots:
    void on_parameter_clicked(Parameters::Parameter* param, QPoint pos);
    void on_deleteNode();
    void on_displayImage();
    void on_plotData();
    bool eventFilter(QObject *object, QEvent *event);
private:
    Parameters::Parameter* currentParam;
    QGraphicsProxyWidget* currentWidget;
    QPoint mousePressPosition;
    std::map<ObjectId, QGraphicsProxyWidget*> widgetMap;
    std::map<EagleLib::IDataStream*, QGraphicsProxyWidget*> dataStreamWidget;
    bool resize = false;
    int resizeGrabSize;
    QPointF grabPoint;
    int corner;
    std::map<QGraphicsProxyWidget*, QGraphicsLineItem*> parentLineMap;
    QMenu* rightClickMenu;
    QList<QAction*> actions;
};

