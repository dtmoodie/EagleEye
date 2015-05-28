#include "NodeListDialog.h"
#include "ui_nodelistdialog.h"
#include "Manager.h"
#include "QListWidgetItem"
NodeListDialog::NodeListDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::NodeListDialog)
{
    ui->setupUi(this);
    update();
}


NodeListDialog::~NodeListDialog()
{
    delete ui;
}
void
NodeListDialog::update()
{
    ui->NodeList->clear();
    auto nodes = EagleLib::NodeManager::getInstance().getConstructableNodes();
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        ui->NodeList->addItem(QString::fromStdString(nodes[i]));
    }
}

void
NodeListDialog::show()
{
    // Update node list with any newly populated nodes

    // Show list
    QDialog::show();
}

void NodeListDialog::on_pushButton_clicked()
{
    EagleLib::Node::Ptr node = EagleLib::NodeManager::getInstance().addNode(ui->NodeList->currentItem()->text().toStdString());
    emit nodeConstructed(node);
}

void NodeListDialog::on_pushButton_2_clicked()
{
    hide();
}
