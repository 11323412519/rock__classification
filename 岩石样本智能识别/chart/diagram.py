import matplotlib.pyplot as plt
def loss_chart(trina_loss,test_loss,epoch,path):
    x=list(range(0,epoch))
    plt.title('Result Analysis')
    plt.plot(x, trina_loss, color='y', label='train loss')
    plt.plot(x, test_loss, color='b', label='test loss')
    plt.legend()  # 显示图例
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("{}\loss.jpg".format(path))
    plt.clf()

def acc_chart(trina_acc,test_acc,epoch,path):
    x = list(range(0, epoch))
    plt.title('Result Analysis')
    plt.plot(x, trina_acc, color='y', label='train acc')
    plt.plot(x, test_acc, color='b', label='test acc')
    plt.legend()  # 显示图例
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.savefig("{}\Accuracy.jpg".format(path))
    plt.clf()


