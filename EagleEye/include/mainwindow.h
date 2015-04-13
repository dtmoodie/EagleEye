#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <nodes/Node.h>
#include <Manager.h>
#include <qtimer.h>
#include "NodeListDialog.h"
#include <qgraphicsscene.h>
#include <qgraphicsview.h>
#include "NodeView.h"
#include <qlist.h>
#include <vector>
#include <boost/thread.hpp>
#include "cornergrabber.h"

namespace Ui {
class MainWindow;
}

class WidgetResizer: public QGraphicsItem
{
public:
    WidgetResizer(QGraphicsScene *scene_ = nullptr);
    bool sceneEventFilter(QGraphicsItem *watched, QEvent *event);

    void setWidget(QGraphicsProxyWidget *widget);
private:
    std::vector<CornerGrabber*> corners;
    QGraphicsScene* scene;
    QGraphicsProxyWidget* currentWidget;
    virtual void mouseMoveEvent ( QGraphicsSceneMouseEvent * event );
    virtual void mouseMoveEvent(QGraphicsSceneDragDropEvent *event);
    virtual void mousePressEvent (QGraphicsSceneMouseEvent * event );
    virtual void mousePressEvent(QGraphicsSceneDragDropEvent *event);
    virtual void mouseReleaseEvent (QGraphicsSceneMouseEvent * event );
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
	void process();
	QList<EagleLib::Node*> getParentNodes();
private slots:
    void on_pushButton_clicked();
    void onTimeout();
	void onNodeAdd(EagleLib::Node* node);
	void onSelectionChanged(QGraphicsProxyWidget* widget);


private:
    void onError(const std::string& error);
    void onStatus(const std::string& status);
    Ui::MainWindow *ui;
    QTimer* fileMonitorTimer;
    NodeListDialog* nodeListDialog;
	QGraphicsScene* nodeGraph;
	NodeView*	nodeGraphView;
	QGraphicsProxyWidget* currentSelectedNodeWidget;
	ObjectId currentNodeId;
	std::vector<ObjectId> parentList;
	boost::thread processingThread;
	bool quit;
    //WidgetResizer* resizer;
};




#endif // MAINWINDOW_H
