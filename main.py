import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pyfaidx import Fasta

from config import *
from utils import set_seed, load_genes, load_expression
from data_loader import GeneExpressionDataset
from models import ExpressionCNN, MLP, LSTMModel
from train_eval import train_epoch, evaluate, compute_metrics, plot_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=WINDOW, help='Half-window size')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--n_tissues', type=int, default=DEFAULT_N_TISSUES, help='Number of tissues to use')
    parser.add_argument('--save_model', type=str, default=SAVE_MODEL_PATH)
    parser.add_argument('--curr_model', type=str, default=CURRENT_MODEL_PATH)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    print("=" * 50)
    print("Загрузка аннотации генов...")
    all_chroms = TRAIN_CHROMS + VAL_CHROMS + TEST_CHROMS
    gene_df = load_genes(GTF_FILE, chrom_list=all_chroms, gene_type='protein_coding')
    print(f"Загружено {len(gene_df)} генов.")

    print("Загрузка данных экспрессии...")
    expr_df = load_expression(EXPR_FILE, n_tissues=args.n_tissues)
    tissue_names = expr_df.columns.tolist()
    print(f"Размер экспрессии: {expr_df.shape} (гены x ткани)")

    print("Инициализация FASTA...")
    fasta = Fasta(FASTA_FILE)

    print("Создание полного датасета...")
    full_dataset = GeneExpressionDataset(gene_df, expr_df, fasta, args.window)
    print(f"Всего валидных генов: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("Нет данных. Проверьте пути к файлам.")
        return

    chrom_for_idx = [gene_df.loc[idx, 'chrom'] for idx in full_dataset.indices]
    train_indices = [i for i, chrom in enumerate(chrom_for_idx) if chrom in TRAIN_CHROMS]
    val_indices   = [i for i, chrom in enumerate(chrom_for_idx) if chrom in VAL_CHROMS]
    test_indices  = [i for i, chrom in enumerate(chrom_for_idx) if chrom in TEST_CHROMS]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)
    test_dataset  = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    input_length = 2 * args.window + 1
    n_tissues = expr_df.shape[1]
    model_cnn = ExpressionCNN(input_length, n_tissues).to(DEVICE)
    model_mlp = MLP(4 * input_length, n_tissues).to(DEVICE)
    model_lstm = LSTMModel(4 * input_length, n_tissues).to(DEVICE)  # input_size=4 (нуклеотиды)

    models = [model_mlp, model_cnn, model_lstm]
    model_names = ['MLP', 'CNN', 'LSTM']

    best_val_loss = float('inf')
    best_model_idx = 0

    for idx, model in enumerate(models):
        print(f"\nОбучение модели: {model_names[idx]}")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        curr_best_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step(val_loss)

            print(f"Epoch {epoch:2d}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_idx = idx
                torch.save(model.state_dict(), args.save_model)
                print(f"  -> лучшая модель сохранена (val loss улучшился)")
            if val_loss < curr_best_loss:
                curr_best_loss = val_loss
                torch.save(model.state_dict(), args.curr_model)
                print(f"  -> текущая лучшая модель сохранена")

        model.load_state_dict(torch.load(args.curr_model))
        test_loss, preds, targets = evaluate(model, test_loader, criterion, DEVICE)
        print(f"\nТест модели {model_names[idx]}: test loss = {test_loss:.4f}")
        mean_r, mean_rmse, r_list, rmse_list = compute_metrics(preds, targets)
        print(f"Средняя корреляция Пирсона: {mean_r:.3f}")
        print(f"Средняя RMSE: {mean_rmse:.3f}")
        for i, (tissue, r, rmse) in enumerate(zip(tissue_names, r_list, rmse_list)):
            if i < 10:
                print(f"  {tissue[:30]:30} R = {r:.3f}, RMSE = {rmse:.3f}")

    print(f"\nЛучшая модель: {model_names[best_model_idx]}")
    best_model = models[best_model_idx]
    best_model.load_state_dict(torch.load(args.save_model))
    test_loss, preds, targets = evaluate(best_model, test_loader, criterion, DEVICE)
    print(f"Test loss: {test_loss:.4f}")
    mean_r, mean_rmse, r_list, rmse_list = compute_metrics(preds, targets)
    print(f"Средняя корреляция Пирсона: {mean_r:.3f}")
    print(f"Средняя RMSE: {mean_rmse:.3f}")

    plot_results(preds, targets, tissue_names, save_path='test_scatter.png')
    print("График сохранён в test_scatter.png")

if __name__ == '__main__':
    main()