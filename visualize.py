import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class GetLocal:
    def __init__(self):
        self.cache = {}

    def clear(self):
        self.cache.clear()

get_local = GetLocal()
def visualize_heads(writer, att_map, cols, step, num):
    to_shows = []
    batch_num = att_map.shape[0]
    head_num = att_map.shape[1]
    # att_map = att_map.squeeze()
    for i in range(batch_num):
        for j in range(head_num):
            to_shows.append((att_map[i][j], f'Batch {i} Head {j}'))
        average_att_map = att_map[i].mean(axis=0)
        to_shows.append((average_att_map, f'Batch {i} Head Average'))

    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])

    writer.add_figure("attention_{}".format(num), fig, step)
    plt.show()


def visualize_segmentation(original_image, predicted_mask, true_mask, title, writer, step):
    """
    可视化分割结果，包括原始图像、预测掩码和真实掩码。

    参数:
        original_image (torch.Tensor): 原始图像
        predicted_mask (torch.Tensor): 预测的分割掩码
        true_mask (torch.Tensor): 真实的分割掩码
        title (str): 图像标题
        writer (SummaryWriter): TensorBoard的SummaryWriter对象
        step (int): 当前步数，用于TensorBoard
    """
    # 将张量从GPU移动到CPU并转换为numpy数组
    original_image = original_image.detach().cpu().numpy()
    predicted_mask = predicted_mask.detach().cpu().numpy()
    true_mask = true_mask.detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(predicted_mask, cmap='gray')
    axs[1].set_title('Predicted Mask')
    axs[1].axis('off')

    axs[2].imshow(true_mask, cmap='gray')
    axs[2].set_title('True Mask')
    axs[2].axis('off')

    plt.suptitle(title)

    # 使用TensorBoard记录图像
    writer.add_figure(title, fig, step)

    plt.show()

# 示例调用
# visualize_segmentation(original_image.cuda(), predicted_mask.cuda(), true_mask.cuda(), "Segmentation Result", writer_visualize, visualize_step)