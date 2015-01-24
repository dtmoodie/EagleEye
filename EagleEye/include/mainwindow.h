#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <nodes/Root.h>

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
    void onStatus(std::string& status);
    Ui::MainWindow *ui;
    EagleLib::Root rootNode;
};

#endif // MAINWINDOW_H