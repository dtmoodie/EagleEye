#pragma once
#include <qgraphicsview.h>
#include <qevent.h>
#include <qgraphicsproxywidget.h>
#include <nodes/Node.h>
#include <QNodeWidget.h>
#include <QMenu>
class NodeView : public QGraphicsView
{
	Q_OBJECT
public:

    NodeView(QWidget* parent = 0);

    NodeView(QGraphicsScene *scene, QWidget *parent = 0);
    void addWidget(QGraphicsProxyWidget * widget, ObjectId id);
    QGraphicsProxyWidget* getWidget(ObjectId id);
    void mousePressEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);
    QGraphicsLineItem* drawLine2Parent(QGraphicsProxyWidget* child);

signals:
	void selectionChanged(QGraphicsProxyWidget* widget);
    void stopThread();
    void startThread();
    void widgetDeleted(QNodeWidget*);
private slots:
    void on_actionSelect(QAction* action);
private:
	QGraphicsProxyWidget* currentWidget;
	QPoint mousePressPosition;
	std::map<ObjectId, QGraphicsProxyWidget*> widgetMap;
    bool resize = false;
    int resizeGrabSize;
    QPointF grabPoint;
    int corner;
    std::map<QGraphicsProxyWidget*, QGraphicsLineItem*> parentLineMap;
    QMenu* rightClickMenu;
    QList<QAction*> actions;
};

