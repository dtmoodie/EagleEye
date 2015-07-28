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
    void wheelEvent(QWheelEvent* event);
    QGraphicsLineItem* drawLine2Parent(QGraphicsProxyWidget* child);
    QGraphicsProxyWidget* getParent(EagleLib::Node::Ptr child);

signals:
	void selectionChanged(QGraphicsProxyWidget* widget);
    void stopThread();
    void startThread();
    void widgetDeleted(QNodeWidget*);
    void plotData(Parameters::Parameter::Ptr param);
    void displayImage(Parameters::Parameter::Ptr param);
private slots:
    void on_parameter_clicked(Parameters::Parameter::Ptr param, QPoint pos);
    void on_deleteNode();
    void on_displayImage();
    void on_plotData();
    bool eventFilter(QObject *object, QEvent *event);
private:
    Parameters::Parameter::Ptr currentParam;
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

