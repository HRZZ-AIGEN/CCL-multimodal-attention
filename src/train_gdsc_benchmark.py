import dataclasses
import shutil
from abc import ABC
from pathlib import Path
from pprint import pformat
from time import time
from typing import Dict, Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multimodal_attention_network import MMAtt
from torch.optim import Adam
from torch.utils.data import DataLoader
from create_graphs import collate, PairDatasetBenchmark
from train_test_split import create_gdsc_split
import click

root = Path(__file__).resolve().parents[1].absolute()

@dataclasses.dataclass(frozen=True)
class Conf:
    gpus: int = 2
    seed: int = 42
    use_16bit: bool = False
    save_dir = '{}/models/'.format(root)
    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 300
    ckpt_path: Optional[str] = None
    reduce_lr: Optional[bool] = False

    def to_hparams(self) -> Dict:
        excludes = [
            'ckpt_path',
            'reduce_lr',
        ]
        return {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if k not in excludes
        }

    def __str__(self):
        return pformat(dataclasses.asdict(self))


class MultimodalAttentionNet(pl.LightningModule, ABC):
    def __init__(
            self,
            hparams,
            data_dir: Optional[Path],
            ppi_depth=3,
            reduce_lr: Optional[bool] = True,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.reduce_lr = reduce_lr
        self.data_dir = data_dir
        self.model = MMAtt(ppi_depth=ppi_depth)
        pl.seed_everything(hparams['seed'])

    def forward(self, x, adj_mat, dist_mat, mask, x_ppi, ppi_edge_index, ppi_batch):
        out, attention_ppi = self.model(x, adj_mat, dist_mat, mask, x_ppi, ppi_edge_index, ppi_batch)
        return out, attention_ppi

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("loss", metrics.get("loss"))
        return metrics.get("loss")

    def validation_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)
        log_metrics = {
            'val_loss': metrics.get("loss")
        }
        self.log('val_loss',
                 metrics.get("loss"),
                 prog_bar=False, on_step=True)
        self.log_dict(log_metrics, on_step=False,
                      on_epoch=True, prog_bar=True)
        return {
            "predictions": metrics.get("predictions"),
            "targets": metrics.get("targets"),
        }

    def validation_epoch_end(self, outputs):
        predictions = torch.cat([x.get('predictions') for x in outputs], 0)
        targets = torch.cat([x.get('targets') for x in outputs], 0)

        rmsefn = torch.nn.MSELoss()
        RMSE = torch.sqrt(rmsefn(predictions, targets))

        log_metrics = {
            'val_RMSE': RMSE,
        }
        self.log_dict(log_metrics)

    def test_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)
        return {
            "predictions": metrics.get("predictions"),
            "attention_ppi": metrics.get("attention_ppi"),
            "targets": metrics.get("targets")
        }

    def test_epoch_end(self, outputs):
        predictions = torch.cat([x.get('predictions') for x in outputs], 0)
        target = torch.cat([x.get('targets') for x in outputs], 0)
        mean_squared_error = torch.nn.MSELoss()
        mse = mean_squared_error(predictions, target)
        rmse = torch.sqrt(mean_squared_error(predictions, target))

        # calculate R2
        r2 = 1 - (torch.sum((predictions-target)**2) / torch.sum((target - torch.mean(target))**2))

        log_metrics = {
            'rmse': rmse,
            'mse': mse,
            'r2': r2,
        }
        self.log_dict(log_metrics)

    def shared_step(self, batch, batch_idx):
        adj_mat, dist_mat, x = batch[0]
        x_ppi = batch[1].x
        ppi_edge_index = batch[1].edge_index
        ppi_batch = batch[1].batch
        mask = torch.sum(torch.abs(x), dim=-1) != 0
        y_hat, attention_ppi = self.forward(
            x,
            adj_mat,
            dist_mat,
            mask,
            x_ppi,
            ppi_edge_index,
            ppi_batch,
        )
        y_hat = y_hat.squeeze(-1)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_hat, (batch[2]))

        return {
            'loss': loss,
            'attention_ppi': attention_ppi,
            'predictions': y_hat,
            'targets': batch[2],
        }

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
        )

        sched = {
            'scheduler': ReduceLROnPlateau(
                opt,
                mode='min',
                patience=15,
                factor=0.5,
            ),
            'monitor': 'val_loss'
        }

        if self.reduce_lr is False:
            return [opt]

        return [opt], [sched]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        version = self.trainer.logger.version[-10:]
        items["v_num"] = version
        return items

    def train_dataloader(self):
        train_dataset = PairDatasetBenchmark(self.data_dir / 'train.csv')

        return DataLoader(train_dataset,
                          self.hparams.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True,
                          pin_memory=True,
                          collate_fn=collate)

    def val_dataloader(self):
        val_dataset = PairDatasetBenchmark(self.data_dir / 'val.csv')

        return DataLoader(val_dataset,
                          self.hparams.batch_size,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True,
                          collate_fn=collate)

    def test_dataloader(self):
        test_dataset = PairDatasetBenchmark(self.data_dir / 'test.csv')
        return DataLoader(test_dataset,
                          self.hparams.batch_size,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True,
                          collate_fn=collate)


@click.command()
@click.option('--split', default='random', type=click.STRING)
@click.option('--seed', default=42, help='Random seed')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=300, help='Number of epochs')
@click.option('--gpu', default=0, help='CUDA GPU order')
@click.option('--checkpoint', default=None, help='Resume training from checkpoint', type=click.STRING)
def main(split, batch_size, epochs, seed, gpu, checkpoint):
    conf = Conf(
        lr=1e-4,
        batch_size=batch_size,
        epochs=epochs,
        reduce_lr=True,
        ckpt_path=checkpoint
    )
    data_dir = Path(create_gdsc_split(split))  # data seed is 42

    model = MultimodalAttentionNet(
        conf.to_hparams(),
        data_dir=data_dir,
        reduce_lr=conf.reduce_lr,
    )

    logger = TensorBoardLogger(
        conf.save_dir,
        name='gdsc_benchmark_{}_{}'.format(split, seed),
        version='{}'.format(str(int(time()))),
    )

    # Copy this script and all files used in training
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(Path(__file__), log_dir)
    shutil.copy(Path(root / "src/train_gdsc_benchmark.py"), log_dir)
    shutil.copy(Path(root / "src/create_graphs.py"), log_dir)
    shutil.copy(Path(root / "src/multimodal_attention_network.py"), log_dir)
    shutil.copy(Path(root / "src/train_test_split.py"), log_dir)

    early_stop_callback = EarlyStopping(monitor='val_loss_epoch',
                                        min_delta=0.00,
                                        mode='min',
                                        patience=25,
                                        verbose=False)
    print("Starting training")
    trainer = pl.Trainer(
        max_epochs=conf.epochs,
        gpus=[gpu],  # [0]
        logger=logger,
        resume_from_checkpoint=conf.ckpt_path,  # load from checkpoint instead of resume
        weights_summary='top',
        callbacks=[early_stop_callback],
        checkpoint_callback=ModelCheckpoint(
            dirpath=(logger.log_dir + '/checkpoint/'),
            monitor='val_loss_epoch',
            mode='min',
            save_top_k=1,
        ),
        deterministic=True,
        auto_lr_find=False,
    )
    trainer.fit(model)
    results = trainer.test()
    results_path = Path(root / "results")
    mse = round(results[0]['mse'], 3)
    rmse = round(results[0]['rmse'], 3)
    r2 = round(results[0]['r2'], 3)
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "Benchmark_results.txt", "w") as file:
            file.write("Benchmark results")
            file.write("\n")

    results = {'Test MSE': mse,
               'Test RMSE': rmse,
               'Test R2': r2}
    version = {'version': logger.version}
    results = {logger.name: [results, version]}
    with open(results_path / "Benchmark_results.txt", "a") as file:
        print(results, file=file)
        file.write("\n")

if __name__ == '__main__':
    main()