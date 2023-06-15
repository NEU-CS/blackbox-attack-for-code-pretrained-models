# train.pkl 格式: text:代码,label:标签 90102条数据，包含原有的训练集
# train_adv.pkl 包含原有训练集和对抗样本
# adv.pkl 只包含对抗样本
# valid.pkl 验证集
# test.pkl 测试集

# adversaial training
使用train_adv.pkl用作训练集
valid.pkl用作验证集


# adversarial fine-tuning
使用train.pkl用作训练集
valid.pkl用作验证集
当eval_acc上升时，使用adv.pkl进行fine-tuning