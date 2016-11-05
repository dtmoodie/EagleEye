#ifndef NODELISTDIALOG_H
#define NODELISTDIALOG_H

#include <QDialog>
#include "EagleLib/Nodes/Node.h"
#include <shared_ptr.hpp>



namespace Ui {
class NodeListDialog;
}

class NodeListDialog : public QDialog, public mo::IMetaObject
{
    Q_OBJECT

public:
    explicit NodeListDialog(QWidget *parent = 0);
    void update();
    void show();
    ~NodeListDialog();

    MO_BEGIN(NodeListDialog)
        MO_SIGNAL(void, add_node, std::string);
    MO_END

signals:
    void nodeConstructed(EagleLib::Nodes::Node::Ptr node);
private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::NodeListDialog *ui;
    
};

#endif // NODELISTDIALOG_H
