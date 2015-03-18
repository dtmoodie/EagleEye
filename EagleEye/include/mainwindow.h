#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <nodes/Root.h>
#include <Manager.h>
#include <qtimer.h>
#include "NodeListDialog.h"
#include <qgraphicsscene.h>
#include <qgraphicsview.h>
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
private slots:
    void on_pushButton_clicked();
    void onTimeout();
	void onNodeAdd(EagleLib::Node* node);

private:
    void onError(const std::string& error);
    void onStatus(const std::string& status);
    Ui::MainWindow *ui;
    QTimer* fileMonitorTimer;
    NodeListDialog* nodeListDialog;
	QGraphicsScene* nodeGraph;
	QGraphicsView*	nodeGraphView;
};

#endif // MAINWINDOW_H
