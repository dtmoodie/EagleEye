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

namespace Ui {
class MainWindow;
}


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void oglDisplay(cv::cuda::GpuMat img, EagleLib::Node *node);
	QList<EagleLib::Node*> getParentNodes();
private slots:
    void on_pushButton_clicked();
    void onTimeout();
	void onNodeAdd(EagleLib::Node* node);
	void onSelectionChanged(QGraphicsProxyWidget* widget);
    void log(QString message);
    void onOGLDisplay(std::string name, cv::cuda::GpuMat img);
signals:
    void eLog(QString message);
    void displayImage(std::string name, cv::cuda::GpuMat img);

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
    boost::mutex parentMtx;
    std::vector<QNodeWidget*> widgets;
	boost::thread processingThread;
	bool quit;
};




#endif // MAINWINDOW_H
