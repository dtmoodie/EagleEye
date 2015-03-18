#include "include/NodeListDialog.h"
#include "ui_nodelistdialog.h"

NodeListDialog::NodeListDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::NodeListDialog)
{
    ui->setupUi(this);
}

NodeListDialog::~NodeListDialog()
{
    delete ui;
}
