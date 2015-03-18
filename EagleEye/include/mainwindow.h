#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <nodes/Root.h>
#include <Manager.h>
#include <qtimer.h>
#include "NodeListDialog.h"
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void show();
private slots:
    void on_pushButton_clicked();
    void onTimeout();

private:
    void onError(const std::string& error);
    void onStatus(const std::string& status);
    Ui::MainWindow *ui;
    QTimer* fileMonitorTimer;
    NodeListDialog* nodeListDialog;
};

#endif // MAINWINDOW_H
