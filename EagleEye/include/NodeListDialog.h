#ifndef NODELISTDIALOG_H
#define NODELISTDIALOG_H

#include <QDialog>
#include "EagleLib/Nodes/Node.h"
#include <shared_ptr.hpp>

#include <signals/signaler.h>

namespace Ui {
class NodeListDialog;
}

class NodeListDialog : public QDialog, public Signals::signaler
{
    Q_OBJECT

public:
    explicit NodeListDialog(QWidget *parent = 0);
    void update();
    void show();
    ~NodeListDialog();

    SIGNALS_BEGIN(NodeListDialog)
        SIG_SEND(add_node, std::string);
    SIGNALS_END

signals:
    void nodeConstructed(EagleLib::Nodes::Node::Ptr node);
private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::NodeListDialog *ui;
    
};

#endif // NODELISTDIALOG_H
