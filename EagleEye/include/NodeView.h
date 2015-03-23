#pragma once
#include <qgraphicsview.h>
#include <qevent.h>
#include <qgraphicsproxywidget.h>
#include <nodes/Node.h>
class NodeView : public QGraphicsView
{
	Q_OBJECT
public:

	NodeView(QWidget* parent = 0) :
		QGraphicsView(parent), currentWidget(nullptr)
	{}

	NodeView(QGraphicsScene *scene, QWidget *parent = 0):
		QGraphicsView(scene, parent), currentWidget(nullptr)
	{}
	void addWidget(QGraphicsProxyWidget * widget, ObjectId id)
	{
		widgetMap[id] = widget;
	}
	QGraphicsProxyWidget* getWidget(ObjectId id)
	{
		auto itr = widgetMap.find(id);
		if (itr != widgetMap.end())
			return itr->second;
		else return nullptr;
	}
	void mousePressEvent(QMouseEvent* event)
	{
		if (QGraphicsItem* item = itemAt(event->pos().x(), event->pos().y()))
		{
			if (QGraphicsProxyWidget* widget = dynamic_cast<QGraphicsProxyWidget*>(item))
			{
				mousePressPosition = event->pos();
				currentWidget = widget;
				emit selectionChanged(widget);
			}
		}else
		{
			QGraphicsView::mousePressEvent(event);
		}
	}
	void mouseMoveEvent(QMouseEvent* event)
	{
		if (currentWidget)
		{
			QPointF pos = currentWidget->pos();
			pos += event->pos() - mousePressPosition;
			currentWidget->setPos(pos);
			mousePressPosition = event->pos();
		}
		QGraphicsView::mouseMoveEvent(event);
	}
	void mouseReleaseEvent(QMouseEvent* event)
	{
		currentWidget = nullptr;
		QGraphicsView::mouseReleaseEvent(event);
	}
signals:
	void selectionChanged(QGraphicsProxyWidget* widget);
	
private:
	QGraphicsProxyWidget* currentWidget;
	QPoint mousePressPosition;
	std::map<ObjectId, QGraphicsProxyWidget*> widgetMap;
};

