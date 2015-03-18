#ifndef NODELISTDIALOG_H
#define NODELISTDIALOG_H

#include <QDialog>

namespace Ui {
class NodeListDialog;
}

class NodeListDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NodeListDialog(QWidget *parent = 0);
    ~NodeListDialog();

private:
    Ui::NodeListDialog *ui;
};

#endif // NODELISTDIALOG_H
