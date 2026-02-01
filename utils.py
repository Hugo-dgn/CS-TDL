import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

def train(model_class, f, mu, n,
                    params_count, epochs, batch_size, lr,
                    device=None):
    model = model_class(n, params_count).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_list = []
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        x = mu(n, batch_size).to(device)
        y = f(x).to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    return loss_list



def compare_models(model1_class, model2_class, f, mu, n,
                   params_count, epochs, batch_size, lr,
                   runs, device=None, verbose=True):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_loss1 = []
    all_loss2 = []

    for run in tqdm(range(runs), disable=not verbose):
        loss1 = train(
            model1_class, f, mu, n,
            params_count, epochs, batch_size, lr,
            device
        )

        loss2 = train(
            model2_class, f, mu, n,
            params_count, epochs, batch_size, lr,
            device
        )

        all_loss1.append(loss1)
        all_loss2.append(loss2)

    all_loss1 = np.array(all_loss1)
    all_loss2 = np.array(all_loss2)

    stats = {
        "epochs": epochs,
        "runs": runs,
        "n": n,
        "Loss1": all_loss1,
        "Loss2": all_loss2,
    }

    return stats




def plot_stats(stats, path=None):
    epochs = stats["epochs"]
    runs = stats["runs"]
    all_loss1 = stats["Loss1"]
    all_loss2 = stats["Loss2"]
    n = stats["n"]
    plt.figure(figsize=(8,5))
    plt.plot(all_loss1.mean(axis=0), label='1 Layer')
    plt.fill_between(range(epochs),
                     all_loss1.mean(axis=0) - all_loss1.std(axis=0),
                     all_loss1.mean(axis=0) + all_loss1.std(axis=0),
                     alpha=0.2)
    plt.plot(all_loss2.mean(axis=0), label='2 Layers')
    plt.fill_between(range(epochs),
                     all_loss2.mean(axis=0) - all_loss2.std(axis=0),
                     all_loss2.mean(axis=0) + all_loss2.std(axis=0),
                     alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Comparison over {runs} runs for n = {n}')
    plt.legend()
    
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
        
        
def parameter_analysis(model1_class, model2_class, f, mu, n,
                   log_params_count_max, epochs, batch_size, lr,
                   runs):
    
    x = []
    y1 = []
    y2 = []
    
    for k in tqdm(range(4, log_params_count_max+1)):
        params_count = 2**k
        stats = compare_models(model1_class, model2_class, f, mu, n, params_count, epochs, batch_size, lr, runs, verbose=False)
        final_loss1 = stats["Loss1"].mean(axis=0)[-1]
        final_loss2 = stats["Loss2"].mean(axis=0)[-1]
        
        x.append(params_count)
        y1.append(final_loss1)
        y2.append(final_loss2)
    
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    return x, y1, y2

def dimension_analysis(model1_class, model2_class, f, mu, log_max_dim,
                   params_count, epochs, batch_size, lr,
                   runs):
    
    x = []
    y1 = []
    y2 = []
    
    for k in tqdm(range(5, log_max_dim+1)):
        n = 2**k
        stats = compare_models(model1_class, model2_class, f, mu, n, params_count, epochs, batch_size, lr, runs, verbose=False)
        final_loss1 = stats["Loss1"].mean(axis=0)[-1]
        final_loss2 = stats["Loss2"].mean(axis=0)[-1]
        
        x.append(n)
        y1.append(final_loss1)
        y2.append(final_loss2)
    
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    return x, y1, y2

def join_analysis(model1_class, model2_class, f, mu, log_max_dim,
                   log_params_count_max, epochs, batch_size, lr,
                   runs):
    
    x = []
    y1 = []
    y2 = []
    
    for k in tqdm(range(6, log_max_dim+1)):
        n = 2**k
        for l in range(4, log_params_count_max+1):
            params_count = 2**l
            
            stats = compare_models(model1_class, model2_class, f, mu, n, params_count, epochs, batch_size, lr, runs, verbose=False)
            final_loss1 = stats["Loss1"].mean(axis=0)[-1]
            final_loss2 = stats["Loss2"].mean(axis=0)[-1]
            
            x.append((n, params_count))
            y1.append(final_loss1)
            y2.append(final_loss2)
    
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    x = x.reshape(-1, log_params_count_max - 3, 2)
    y1 = y1.reshape(-1, log_params_count_max - 3)
    y2 = y2.reshape(-1, log_params_count_max - 3)

    return x, y1, y2