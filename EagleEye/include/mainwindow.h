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
    void qtDisplay(cv::Mat img, EagleLib::Node *node);
    void onCompileLog(const std::string& msg, int level);
private slots:
    void on_pushButton_clicked();
    void onTimeout();
    void onNodeAdd(EagleLib::Node::Ptr node);
	void onSelectionChanged(QGraphicsProxyWidget* widget);
    void log(QString message);
    void onOGLDisplay(std::string name, cv::cuda::GpuMat img);
    void onQtDisplay(std::string name, cv::Mat img);
    void stopProcessingThread();
    void startProcessingThread();
    void onWidgetDeleted(QNodeWidget* widget);
    void onSaveClicked();
    void onLoadClicked();
    void addNode(EagleLib::Node::Ptr ndoe);
    void updateLines();
signals:
    void eLog(QString message);
    void oglDisplayImage(std::string name, cv::cuda::GpuMat img);
    void qtDisplayImage(std::string name, cv::Mat img);



private:
    void onError(const std::string& error);
    void onStatus(const std::string& status);
    Ui::MainWindow *ui;
    QTimer* fileMonitorTimer;
    NodeListDialog* nodeListDialog;
	QGraphicsScene* nodeGraph;
	NodeView*	nodeGraphView;
	QGraphicsProxyWidget* currentSelectedNodeWidget;
    EagleLib::Node::Ptr currentNode;
    std::vector<EagleLib::Node::Ptr> parentList;
    boost::recursive_mutex parentMtx;
    std::vector<QNodeWidget*> widgets;
	boost::thread processingThread;
	bool quit;
};




#endif // MAINWINDOW_H
