# ATDD Checklist: Story 4.2 Reproducibility & Determinism Verification

**Story ID**: 4.2
**Status**: RED (Targeting Failures)
**Test Strategy**: Integration/System Testing (Verification of NFR-R1)

## 📋 驗收標準與測試映射 (Acceptance Criteria Mapping)

| ID | 驗收標準 (Acceptance Criteria) | 測試文件 (Test File) | 測試級別 (Level) |
|---|---|---|---|
| AC1 | Loss 值在固定種子下的兩次運行中完全相同 (小數點後 6 位) | `tests/test_reproducibility.py` | Integration |
| AC2 | 最終模型權重 (state_dict) 在固定種子下完全相同 (Bit-for-bit) | `tests/test_reproducibility.py` | Integration |
| AC3 | 首次評估的 mAP 值完全一致 | `tests/test_reproducibility.py` | Integration |
| AC4 | 驗證 Trainer 配置了 deterministic=True | `tests/test_reproducibility.py` | Unit/Config |

## 🧪 失敗測試生成 (Failing Tests Generation)

### 1. 測試文件結構

```
tests/
└── test_reproducibility.py  # 主要驗證測試
```

### 2. 測試實施 (預期失敗)

**文件**: `tests/test_reproducibility.py`

此測試將嘗試運行兩個短暫的訓練循環。目前代碼可能缺少 `seed_everything` 的正確調用或 `deterministic` 標誌配置，預計會失敗或產生警告。

```python
import pytest
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from src.models.agch_module import AGCHModule
from src.data.agch_datamodule import AGCHDataModule
# 假設 metrics.py 已經在 Story 4.1 實現
from src.utils.metrics import calculate_mAP

@pytest.mark.integration
def test_training_determinism(tmp_path):
    """
    驗證固定種子下的訓練過程是否完全可重現 (Loss 和 Weights)。
    """
    
    def run_short_training(seed):
        # 設置種子
        seed_everything(seed, workers=True)
        
        # 創建配置 (模擬 hydration)
        cfg = OmegaConf.create({
             "model": {"alpha": 0.1, "beta": 0.1, "gamma": 0.1, "hash_code_len": 12},
             "data": {"data_dir": "data/", "batch_size": 16, "num_workers": 0}, # 使用0 workers避免多進程複雜性
             "trainer": {"max_epochs": 1, "accelerator": "cpu", "devices": 1, "deterministic": True, "logger": False, "enable_checkpointing": False}
        })

        # 初始化數據和模型
        datamodule = AGCHDataModule(data_dir=cfg.data.data_dir, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
        model = AGCHModule(**cfg.model)
        
        # 初始化 Trainer (限制為幾個 steps)
        trainer = Trainer(
            default_root_dir=str(tmp_path),
            limit_train_batches=5,  # 只跑 5 個 batch
            limit_val_batches=0,    # 跳過驗證以加速
            **cfg.trainer
        )
        
        # 運行訓練
        trainer.fit(model, datamodule=datamodule)
        
        # 收集結果
        final_loss = trainer.callback_metrics.get("train/loss_total")
        state_dict = model.state_dict()
        
        return final_loss, state_dict

    # 運行兩次
    loss_1, weights_1 = run_short_training(seed=42)
    loss_2, weights_2 = run_short_training(seed=42)
    
    # 斷言 Loss 相同
    assert torch.allclose(loss_1, loss_2, atol=1e-6), f"Loss mismatch: {loss_1} != {loss_2}"
    
    # 斷言 Weights 相同
    for key in weights_1:
         assert torch.equal(weights_1[key], weights_2[key]), f"Weight mismatch in validation: {key}"

@pytest.mark.unit
def test_trainer_configuration_determinism():
    """
    驗證默認配置中是否啟用了 deterministic 標誌。
    """
    # 加載實際的 trainer 配置
    with hydra.initialize(version_base=None, config_path="../../configs/trainer"):
        cfg = hydra.compose(config_name="default")
        
    assert cfg.deterministic is True, "Trainer config must have 'deterministic: True'"
    assert cfg.benchmark is False, "Trainer config must have 'benchmark: False' for reproducibility"

```

## 🛠️ 實施清單 (Implementation Checklist)

### 紅燈階段 (RED)
- [x] 生成 `tests/test_reproducibility.py`
- [ ] 運行測試確認失敗 (由用戶/DEV執行)
    - 預期失敗原因: `default.yaml` 中可能默認 `deterministic=False`，或 `seed_everything` 位置不正確。

### 綠燈階段 (GREEN - DEV 任務)
- [ ] 修改 `configs/trainer/default.yaml`: 設置 `deterministic: True`, `benchmark: False`
- [ ] 審查 `src/train.py`: 確保 `seed_everything` 在所有初始化之前調用
- [ ] 運行測試確認通過

### 重構階段 (REFACTOR)
- [ ] 確保測試使用的 Mock 數據不會引入外部依賴
- [ ] 優化測試執行速度 (使用 CPU 或極小模型)

## 🚀 執行命令

```bash
# 運行再現性測試
pytest tests/test_reproducibility.py
```
