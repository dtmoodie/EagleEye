#ifndef NODELISTDIALOG_H
#define NODELISTDIALOG_H

#include <QDialog>

namespace Ui {
class NodeListDialog;
}
class Node;
class NodeListDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NodeListDialog(QWidget *parent = 0);
    void show();
    ~NodeListDialog();
signals:
    void nodeConstructed(Node* node);
private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::NodeListDialog *ui;
};

#endif // NODELISTDIALOG_H
