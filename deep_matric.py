#%%
import os, json, sys, glob
import torch, torchvision
import numpy as np
import random


from torchvision.models._utils import IntermediateLayerGetter
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


#%%


deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
model = IntermediateLayerGetter(deeplabv3, {"classifier" : "out"})


# %%

class LeafDataset(object):

    def __init__(self, root, train):
        self.root = root
        self.train = train
        # load all image files, sorting them to
        # ensure that they are aligned

    def transform(self, image, mask):

        size = (512, 512)
        size = (256, 256)

        image = T.Resize(size=size, interpolation=Image.BILINEAR)(image)
        mask = T.Resize(size=size, interpolation=Image.NEAREST)(mask)

        # angle = random.randint(-180, 180)
        # image = TF.rotate(image, angle, resample= Image.BILINEAR)
        # mask = TF.rotate(mask, angle, resample=Image.NEAREST)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def gen_basename(self, idx):
        if idx < 10:
            return f"ara2012_plant00{idx}"
        elif 10 <= idx < 100:
            return f"ara2012_plant0{idx}"
        else:
            return f"ara2012_plant{idx}"

    def __getitem__(self, idx):
        basename = self.gen_basename(idx+1)
        img_path = os.path.join(self.root, f"{basename}_rgb.png")
        mask_path = os.path.join(self.root, f"{basename}_label.png")
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # first id is the background, so remove it
        # suppose all instances are not crowd
        if self.train:
            img, mask = self.transform(img, mask)
        # to_tensorは-1~1に正規化するので,maskはto_tensorを使わずにTensorにしてる。
        img = TF.to_tensor(img)
        mask = torch.Tensor(np.array(mask, dtype=np.uint8))
        return img, mask

    def __len__(self):
        return 120


class MetricModel(nn.Module):

    def __init__(self, seg_model, embedding_d = 16):
        super(MetricModel, self).__init__()
        seg_model_children = list(seg_model.children())
        self.backbone = seg_model_children[0]
        self.middle = nn.Sequential(*list(seg_model_children[1].children())[:-1])
        self.last_conv = nn.Conv2d(256, embedding_d, 1, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        x = self.backbone(x)["out"]
        x = self.middle(x)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False, )
        return x

#%%

class MetricLoss(nn.Module):

    def __init__(self,  alpha=1, beta=1, gamma=0.001, variance_theta=0.5, distance_theta=1.5):
        super(MetricLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.variance_theta = variance_theta
        self.distance_theta = distance_theta


    def forward(self, embeddings, g_truths, writer=None, iterate=None):
        batch_size = len(embeddings)
        losses = torch.zeros(batch_size)
        variance_losses = torch.zeros(batch_size)
        distance_losses = torch.zeros(batch_size)
        regularization_losses = torch.zeros(batch_size)

        for batch_i, zip_input in enumerate(zip(embeddings, g_truths)):
            _loss = 0
            embedding, g_truth = zip_input
            variance_loss, cluster_means = self.get_variance_loss(embedding, g_truth)
            distance_loss = self.get_distance_loss(cluster_means)
            regularization_loss = torch.mean(torch.norm(cluster_means, dim=1))
            loss = self.alpha * variance_loss + self.beta * distance_loss + self.gamma * regularization_loss
            losses[batch_i] = loss
            variance_losses[batch_i] = variance_loss
            distance_losses[batch_i] = distance_loss
            regularization_losses[batch_i] = regularization_loss
        if iterate == 22:
            print("check")
            print("check")

        if writer:
            writer.add_scalar('Loss/train/variance_loss', torch.mean(variance_losses), iterate)
            writer.add_scalar('Loss/train/distance_loss', torch.mean(distance_losses), iterate)
            writer.add_scalar('Loss/train/regularization_loss', torch.mean(regularization_losses), iterate)
            writer.add_scalar('Loss/train/total_loss', torch.mean(losses), iterate)

        return torch.mean(losses)

    def get_variance_loss(self, embedding, g_truth):
        """
            同じクラスがクラス中心nに引き寄せられるようにするロス。
            Args:
                - embedding (torch.Tensor) : (embedinng_d, w, h)のembedding層のoutput
                - g_truths (torch.Tensor) : (1, w, h)の正解データ。各ピクセルには、クラスのe_numが記載されてる。
                TODO : vairance_theta正しい値設定。
        """
        emmbeding_d = embedding.shape[0]
        cluster_num = int(torch.max(g_truth)) + 1
        cluster_means = torch.zeros((cluster_num, emmbeding_d))
        variance_loss = 0
        for cluster_i in range(cluster_num):
            # (-1, 1, h, w) を
            # clusterの中であるclusterのものだけを取得
            # print(f"cluster_i {cluster_i}")

            mask_boolen = g_truth == cluster_i
            masked_embedding = torch.t(embedding[: , mask_boolen]).contiguous()
            masked_embedding_num = len(masked_embedding)
            cluster_mean = torch.mean(masked_embedding, axis=0) # 次元embedding_dとなるクラスタの中心ベクトル
            print(f"-- cluster_mean : {cluster_mean} --")

            mean_difference_vectors = torch.norm((cluster_mean - masked_embedding) , dim=1) - self.variance_theta
            print(f"-- {mean_difference_vectors=} --")

            cluster_loss = torch.sum(torch.clamp(mean_difference_vectors, min=0.0) ** 2) / masked_embedding_num
            variance_loss += cluster_loss
            cluster_means[cluster_i] = cluster_mean
        # import ipdb; ipdb.set_trace()
        variance_loss = variance_loss / cluster_num
        cluster_means = torch.Tensor(cluster_means)
        return variance_loss, cluster_means

    def get_distance_loss(self, cluster_means):
        """
            クラス同士の距離を遠くするロス。
        """
        distance_loss = 0

        matmul = torch.matmul(cluster_means, cluster_means.T)
        cluster_num = len(cluster_means)
        for cluster_a_idx in range(cluster_num):
            for cluster_b_idx in range(cluster_a_idx+1, cluster_num):
                cluster_mean_a = cluster_means[cluster_a_idx]
                cluster_mean_b = cluster_means[cluster_b_idx]
                # print("-- cluster_mean_a, cluster_mean_b --")
                # print(cluster_mean_a, cluster_mean_b)
                tmp_loss = 2 * self.distance_theta - torch.norm(cluster_mean_b - cluster_mean_a)
                distance_loss += torch.clamp(tmp_loss, min=0.0) ** 2
        return distance_loss / (cluster_num * (cluster_num - 1))


def train(model, data_loader, data_loader_test, num_epochs=50, device="cpu"):
    model.to(device)
    writer = SummaryWriter()
    criterion = MetricLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        batch_count = 0
        # train for one epoch, printing every 10 iterations
        for i, (img, mask) in enumerate(data_loader):
            img = img.to(device)
            mask = mask.to(device)
            current_iterate = len(data_loader) * epoch + i
            print("- data loaded - ")
            # update the learning rate
            model.train()
            output = model(img)
            loss = criterion(output, mask, writer=writer, iterate=current_iterate)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.to("cpu").item()
            batch_count += 1
            print(f"\n== iterate : {current_iterate}, loss : {loss.to('cpu').item()} ==")

        for img, mask in data_loader_test:
            model.eval()
            with torch.no_grad():
                output = model(img)
                loss = criterion(output, mask)
                val_loss += loss.to("cpu").item()

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    root = os.path.join("data", "Plant_Phenotyping_Datasets", "Plant", "Ara2012")

    dataset = LeafDataset(root, train=True)
    dataset_test = LeafDataset(root, train=False)

    indices = torch.randperm(len(dataset)).tolist()
    train_num = 100
    dataset = torch.utils.data.Subset(dataset, indices[:train_num])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_num:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
    )


    deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    model = MetricModel(deeplabv3)

    train(model, data_loader, data_loader_test)

if __name__ == "__main__":
    main()
# %%
