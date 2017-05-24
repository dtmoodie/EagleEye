#include "signal_dialog.h"
#include "ui_signal_dialog.h"
#include <MetaObject/object/RelayManager.hpp>
#include <MetaObject/signals/ISignalRelay.hpp>
#include <qinputdialog.h>

signal_dialog::signal_dialog(mo::RelayManager* manager, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::signal_dialog)
{
    ui->setupUi(this);
    ui->treeWidget->setColumnCount(3);
    connect(ui->treeWidget, &QTreeWidget::itemDoubleClicked, this, &signal_dialog::on_item_select);
    update(manager);
}

void signal_dialog::update(mo::RelayManager* manager)
{
    _manager = manager;
    ui->treeWidget->clear();
    auto all_relays = manager->getAllRelays();
    std::map<std::string, QTreeWidgetItem*> top_level_items;
    for(auto& relay : all_relays )
    {
        QTreeWidgetItem* parent = nullptr;
        auto itr = top_level_items.find(relay.second);
        if(itr != top_level_items.end())
        {
            parent = itr->second;
        }else
        {
            parent = new QTreeWidgetItem(ui->treeWidget);
            top_level_items[relay.second] = parent;
            parent->setText(0, QString::fromStdString(relay.second));
        }
        if(parent)
        {
            QTreeWidgetItem* child = new QTreeWidgetItem(parent);
            child->setText(0, QString::fromStdString(relay.second));
            child->setText(2, QString::fromStdString(relay.first->getSignature().name()));
            //child->setText(1, QString::fromStdString(receiver.description));
            //child->setToolTip(0, QString::fromStdString(receiver.tooltip));
        }
    }
    /*auto names = manager->get_signal_names();
    for(auto& name : names)
    {
        if(top_level_items.find(name) == top_level_items.end())
        {
            QTreeWidgetItem* item = new QTreeWidgetItem(ui->treeWidget);
            item->setText(0, QString::fromStdString(name));
            top_level_items[name] = item;
        }
    }*/
}

signal_dialog::~signal_dialog()
{
    delete ui;
}

void signal_dialog::on_item_select(QTreeWidgetItem* item, int column)
{
    /*auto factory = Signals::serialization::text::factory::instance();
    if(item->parent() == nullptr)
    {
        auto all_signals = _manager->get_signals(item->text(0).toStdString());
        for(auto signal : all_signals)
        {
            if(signal->get_signal_type() == mo::TypeInfo(typeid(void(void))))
            {
                (*static_cast<mo::TypeInfo<void(void)>*>(signal))();
            }else
            {
                std::unique_ptr<Signals::serialization::text::serialization_proxy_base> proxy(factory->get_proxy(signal->get_signal_type()));
                QString text = QInputDialog::getText(this, "Signal arguments", "Arguments: ");
                proxy->send(signal, text.toStdString());
            }
        }
    }else
    {
        auto signal = _manager->get_signal_optional(item->parent()->text(0).toStdString(), item->text(2).toStdString());
        if(signal)
        {
            if(signal->get_signal_type() == mo::TypeInfo(typeid(void(void))))
            {
                (*static_cast<mo::TypeInfo<void(void)>*>(signal))();
            }else
            {
                std::unique_ptr<Signals::serialization::text::serialization_proxy_base> proxy(factory->get_proxy(signal->get_signal_type()));
                QString text = QInputDialog::getText(this, "Signal arguments", "Arguments: ");
                proxy->send(signal, text.toStdString());
            }
        }
    }*/
}
