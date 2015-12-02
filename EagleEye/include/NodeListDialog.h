#ifndef NODELISTDIALOG_H
#define NODELISTDIALOG_H

#include <QDialog>
#include "nodes/Node.h"
#include <EagleLib/shared_ptr.hpp>
namespace Ui {
class NodeListDialog;
}

class NodeListDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NodeListDialog(QWidget *parent = 0);
    void update();
    void show();
    ~NodeListDialog();
signals:
    void nodeConstructed(EagleLib::Node::Ptr node);
private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::NodeListDialog *ui;
	
};

#endif // NODELISTDIALOG_H
