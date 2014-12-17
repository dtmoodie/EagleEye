#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "nodes/ParallelStack.h"
#include "nodes/SerialStack.h"

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

private:
    void onError(std::string& error);
    Ui::MainWindow *ui;
    EagleLib::SerialStack baseNode;
};

#endif // MAINWINDOW_H
