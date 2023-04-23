from var import epoch_num, loss_fn, optimizer, lr_scheduler, device, bert_model
from data_preprocess import train_dataloader, dev_dataloader
from train import train, test
train(train_dataloader, bert_model, loss_fn, optimizer, lr_scheduler, epoch_num, device)
test(dev_dataloader, bert_model, device)
print("Done!")
