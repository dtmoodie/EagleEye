#pragma once
#include <qgraphicsview.h>
#include <qevent.h>
#include <qgraphicsproxywidget.h>
#include <Aquila/Nodes/Node.h>
#include <QNodeWidget.h>
#include <QMenu>
#include <shared_ptr.hpp>
class signal_dialog;
class NodeView : public QGraphicsView
{
    Q_OBJECT
public:

    NodeView(QWidget* parent = 0);

    NodeView(QGraphicsScene *scene, QWidget *parent = 0);
    void addWidget(QGraphicsProxyWidget * widget, ObjectId id);
    void addWidget(QGraphicsProxyWidget* widget, aq::IDataStream* stream_);
    QGraphicsProxyWidget* getWidget(ObjectId id);
    QGraphicsProxyWidget* getWidget(aq::IDataStream* stream_);
    
    void removeWidget(ObjectId id);
    void mousePressEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);
    void wheelEvent(QWheelEvent* event);
    QGraphicsLineItem* drawLine2Parent(QGraphicsProxyWidget* child);
    QGraphicsProxyWidget* getParent(aq::Nodes::Node::Ptr child);
    std::vector<QGraphicsProxyWidget*> getParents(aq::Nodes::Node::Ptr child);
    QGraphicsProxyWidget* getStream(aq::IDataStream* stream_id);

signals:
    void selectionChanged(QGraphicsProxyWidget* widget);
    void stopThread();
    void startThread();
    void widgetDeleted(QNodeWidget*);
    void widgetDeleted(DataStreamWidget*);
    void plotData(mo::IParameter* param);
    void displayImage(mo::IParameter* param);
private slots:
    void on_parameter_clicked(mo::IParameter* param, QPoint pos);
    void on_deleteNode();
    void on_displayImage();
    void on_plotData();
    void on_display_signals();
    bool eventFilter(QObject *object, QEvent *event);
private:
    mo::IParameter* currentParam;
    QGraphicsProxyWidget* currentWidget;
    QPoint mousePressPosition;
    std::map<ObjectId, QGraphicsProxyWidget*> widgetMap;
    std::map<aq::IDataStream*, QGraphicsProxyWidget*> dataStreamWidget;
    bool resize = false;
    int resizeGrabSize;
    QPointF grabPoint;
    int corner;
    std::map<QGraphicsProxyWidget*, QGraphicsLineItem*> parentLineMap;
    QMenu* rightClickMenu;
    QList<QAction*> actions;
    signal_dialog* _signal_dialog;
};

