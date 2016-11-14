#ifndef NODELISTDIALOG_H
#define NODELISTDIALOG_H

#include <QDialog>
#include "EagleLib/Nodes/Node.h"
#include <shared_ptr.hpp>



namespace Ui {
class NodeListDialog;
}

class NodeListDialog : public QDialog
{
    Q_OBJECT
public:
    explicit NodeListDialog(QWidget *parent = 0);
    ~NodeListDialog();
    void update();
    mo::TypedSlot<void()> update_slot;
private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::NodeListDialog *ui;
    mo::TypedSignal<void(std::string)> add_node_signal;
    std::shared_ptr<mo::Connection> connection;
};

#endif // NODELISTDIALOG_H
