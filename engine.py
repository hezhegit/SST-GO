import pickle as pkl
import warnings

import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import evaluation
from graph_data import GoTermDataset, collate_fn

warnings.filterwarnings("ignore")

def train(model: torch.nn.Module, config, task: str):
    # 1. 数据集与 DataLoader
    train_set = GoTermDataset("train", task)
    valid_set = GoTermDataset("val", task)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 2. 模型、优化器、损失函数
    model = model.to(config.device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **config.optimizer
    )
    bce = torch.nn.BCELoss(reduction='none')

    best_eval_loss = float('inf')
    es_counter = 0

    for epoch in range(config.max_epochs):
        # —— 训练阶段 —— #
        model.train()
        train_pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] train", leave=False)
        for data, y_true in train_pbar:
            data = data.to(config.device)
            y_true = y_true.to(config.device)

            optimizer.zero_grad()
            combined, l_esm, l_graph = model(data)  # forward
            # l = model(data)  # forward

            # 主 loss + 辅助 losses
            loss_main = bce(torch.sigmoid(combined), y_true).mean()
            loss_e    = bce(torch.sigmoid(l_esm),    y_true).mean()
            loss_g    = bce(torch.sigmoid(l_graph),  y_true).mean()
            loss = loss_main + 0.3 * (loss_e  + loss_g)
            # loss = bce(torch.sigmoid(l),  y_true).mean()

            loss.backward()
            optimizer.step()

            train_pbar.set_postfix(loss=loss.item())

        # —— 验证阶段 —— #
        model.eval()
        all_preds = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"[Epoch {epoch}] valid", leave=False)
            for data, y_true in val_pbar:
                data = data.to(config.device)
                y_true = y_true.to(config.device)

                combined,_,_ = model(data)
                preds = torch.sigmoid(combined)
                all_preds.append(preds.cpu())

            y_pred_all = torch.cat(all_preds, dim=0)
            y_true_all = valid_set.y_true.float()

            eval_loss = bce(y_pred_all, y_true_all).mean().item()
            aupr = metrics.average_precision_score(
                y_true_all.numpy(),
                y_pred_all.numpy(),
                average="samples"
            )
            fm = fmax(y_true_all.numpy(), y_pred_all.numpy(), nrThresholds=10)

            print(f"Epoch {epoch:02d} | val_loss={eval_loss:.4f} | AUPR={aupr:.4f} | Fmax={fm:.4f}")

        # —— 模型保存与早停 —— #
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            es_counter = 0
            torch.save(model.state_dict(), config.model_save_path + task + ".pt")
            print(f"  → Saved best model (loss {best_eval_loss:.4f})")
        else:
            es_counter += 1
            print(f"  EarlyStopping counter: {es_counter}/5")
            if es_counter > 4:
                torch.save(model.state_dict(), config.model_save_path + task + ".pt")
                print("  → Early stopping triggered.")
                break

    print("Training completed.")


def test(model: torch.nn.Module, config, task: str):
    test_set = GoTermDataset("test", task)
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = model.to(config.device)
    model.load_state_dict(torch.load(config.model_save_path + task + ".pt", map_location=config.device))
    model.eval()

    bce = torch.nn.BCELoss()

    all_preds = []
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="test", leave=False)
        for data, y_true in test_pbar:
            data = data.to(config.device)
            y_true = y_true.to(config.device)

            combined,_,_ = model(data)
            preds = torch.sigmoid(combined)
            all_preds.append(preds.cpu())

    y_pred_all = torch.cat(all_preds, dim=0)
    y_true_all = test_set.y_true.float()

    test_loss = bce(y_pred_all, y_true_all).item()
    # fm = fmax(y_true_all.numpy(), y_pred_all.numpy(), nrThresholds=10)
    # print(f"Test Loss: {test_loss:.4f} | Fmax: {fm:.4f}")

    # 保存结果
    result_path = config.test_result_path + task + ".pt"
    with open(result_path, "wb") as f:
        pkl.dump([y_pred_all.numpy(), y_true_all.numpy()], f)
    print(f"Test results saved to {result_path}")
    print(f"== Start Evaluation ==")

    # load termIC
    with open("/home/415/hz_project/SSGO/data/ic_count.pkl", 'rb') as f:
        ic_count = pkl.load(f)
    ic_count['bp'] = np.where(ic_count['bp'] == 0, 1, ic_count['bp'])
    ic_count['mf'] = np.where(ic_count['mf'] == 0, 1, ic_count['mf'])
    ic_count['cc'] = np.where(ic_count['cc'] == 0, 1, ic_count['cc'])
    termIC = {
        'bp': -np.log2(ic_count['bp'] / 69709),
        'mf': -np.log2(ic_count['mf'] / 69709),
        'cc': -np.log2(ic_count['cc'] / 69709)
    }
    # ================== 读取保存的预测结果 ==================
    with open(result_path, "rb") as f:
        y_pred_all, y_true_all = pkl.load(f)  # y_pred_all: (N, C), y_true_all: (N, C)

    # 直接转换为 numpy 数组（如果还不是的话）
    y_pred_all = np.asarray(y_pred_all, dtype=float)
    y_true_all = np.asarray(y_true_all, dtype=int)

    print(f"Loaded shapes: y_pred: {y_pred_all.shape}, y_true: {y_true_all.shape}")

    # ================== 计算评估指标 ==================
    Fmax_val = evaluation.fmax(y_true_all, y_pred_all, task)
    print(f"[Eval] Fmax: {Fmax_val:.4f}")

    MacroAUPR = evaluation.macro_aupr(y_true_all, y_pred_all, task)
    print(f"[Eval] Macro-AUPR: {MacroAUPR:.4f}")

    Smin_val = evaluation.smin(termIC[task], y_true_all, y_pred_all, task)
    print(f"[Eval] Smin: {Smin_val:.4f}")





def fmax(Ytrue, Ypred, nrThresholds):
    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ff = np.zeros(thresholds.shape)
    pr = np.zeros(thresholds.shape)
    rc = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        thr = np.round(t, 2)
        pr[i], rc[i], ff[i], _ = precision_recall_fscore_support(Ytrue, (Ypred >= t).astype(int), average='samples')

    return np.max(ff)