#include "NodeListDialog.h"
#include "ui_nodelistdialog.h"


#include "QListWidgetItem"
NodeListDialog::NodeListDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::NodeListDialog)
{
    ui->setupUi(this);
    EagleLib::ObjectManager::Instance().RegisterConstructorAddedCallback(boost::bind(&NodeListDialog::update, this));
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
        //ui->NodeList->addItem(QString::fromStdString(nodes[i]));
        auto info  = EagleLib::NodeManager::getInstance().GetNodeInfo(nodes[i]);
        QTreeWidgetItem* parent = nullptr;
        if (info.size())
        {
            for (int j = 0; j < ui->NodeList->topLevelItemCount(); ++j)
            {
                if (ui->NodeList->topLevelItem(j)->text(0) == QString(info[0]))
                {
                    parent = ui->NodeList->topLevelItem(j);
                }
            }
            if (!parent)
            {
                parent = new QTreeWidgetItem(ui->NodeList);
                ui->NodeList->addTopLevelItem(parent);
                parent->setText(0, QString(info[0]));
            }
            for (int k = 1; k < info.size(); ++k)
            {
                bool found = false;
                for (int j = 0; j < parent->childCount(); ++j)
                {
                    if (parent->child(j)->text(0) == QString(info[k]))
                    {
                        found = true;
                        parent = parent->child(j);
                        break;
                    }
                }
                if (!found)
                {
                    auto newParent = new QTreeWidgetItem(parent);
                    newParent->setText(0, QString(info[k]));
                    parent->addChild(newParent);
                    parent = newParent;
                }
            }
            auto node = new QTreeWidgetItem(parent);
            node->setText(0, QString::fromStdString((nodes[i])));
            parent->addChild(node);
        }
        else
        {
            QTreeWidgetItem* item = new QTreeWidgetItem(ui->NodeList);
            item->setText(0, QString::fromStdString(nodes[i]));
            ui->NodeList->addTopLevelItem(item);
        }
    }
    //ui->NodeList->sortItems();

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
    if(ui->NodeList->currentItem())
    {
        
        sig_add_node(ui->NodeList->currentItem()->text(0).toStdString());
        //EagleLib::Nodes::Node::Ptr node = EagleLib::NodeManager::getInstance().addNode(ui->NodeList->currentItem()->text(0).toStdString());
        //if (node != nullptr)
        //{
            //emit nodeConstructed(node);
        //}
        
    }
}

void NodeListDialog::on_pushButton_2_clicked()
{
    hide();
}
