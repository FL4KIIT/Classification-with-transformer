"""
#Датасет IMDB скачан с Kaggle/HF и распакован локально для стабильности

================================================================================

Обоснование выбора датасета и модели:
- Датасет: IMDb (50k обзоров фильмов, бинарный сантимент). Классический,
  сбалансированный, англоязычный.
- Модель: bert-base-uncased. Поддерживает английский, относительно компактна
  (110M параметров), является стандартным выбором для сравнения методов
  fine-tuning. Uncased-версия игнорирует регистр, что полезно для текстов
  с разнородным капитализацией.
- MAX_LENGTH=128:  уменьшает расход памяти.
- Гиперпараметры:
  * Full FT: lr=2e-5 – стандарт для BERT, предотваращает  забывание.
  * Linear Probing: lr=1e-4 / epochs=5 – выше lr для быстрого обучения,
    т.к. обучается только классификатор.
  * LoRA: lr=3e-4 – более высокий lr для адаптеров, r=16 / alpha=32 – хорошо
    зарекомендовавшая себя конфигурация.
- FP16: автоматически включается при наличии GPU, ускоряет и уменьшает память.
================================================================================
"""
import os
import random
import time
import warnings
import zipfile
import csv
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    logging as hf_logging
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# ================= КОНФИГУРАЦИЯ =================
SEED = 42
MODEL_NAME = "bert-base-uncased" #английский датасет. uncased версия лучше подходит для отзывов, где регистр часто не несет смысловой нагрузки,
                                 # а также она немного быстрее и требует меньше памяти, чем cased.
MAX_LENGTH = 128
TRAIN_SUBSET = 2000
EVAL_SUBSET = 500
LOCAL_ZIP_PATH = r"C:\Users\Yaroslav\Downloads\archive.zip"
EXTRACT_DIR = "./data/imdb_extracted"



def fix_random_state(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_local_imdb_from_zip(zip_path: str, extract_dir: str) -> DatasetDict:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f" Архив не найден: {zip_path}")

    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        print(f" Распаковка {zip_path} в {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
    else:
        print(f" Датасет уже распакован в {extract_dir}")

    root = Path(extract_dir)
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if len(dirs) == 1:
        root = dirs[0]

    print(f" Сканирование структуры в: {root}")
    print("    Содержимое:", [p.name for p in root.iterdir()][:10])

    csv_files = list(root.rglob("*.csv"))
    if csv_files:
        print(f" Найден CSV: {csv_files[0].name}. Парсинг...")
        texts, labels = [], []
        with open(csv_files[0], encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            text_col = next((h for h in headers if "review" in h.lower() or "text" in h.lower()), headers[0])
            label_col = next((h for h in headers if "sentiment" in h.lower() or "label" in h.lower()), headers[1])

            for row in reader:
                txt = row.get(text_col, "").strip()
                lbl = row.get(label_col, "").strip().lower()
                if not txt: continue
                if lbl in ("positive", "pos", "1"):
                    labels.append(1)
                elif lbl in ("negative", "neg", "0"):
                    labels.append(0)
                else:
                    continue
                texts.append(txt)

        print(f"   Всего примеров из CSV: {len(texts)}")
        train_txt, test_txt, train_lbl, test_lbl = train_test_split(
            texts, labels, test_size=0.2, random_state=SEED, stratify=labels
        )
        return DatasetDict({
            "train": Dataset.from_dict({"text": train_txt, "label": train_lbl}),
            "test": Dataset.from_dict({"text": test_txt, "label": test_lbl})
        })

    print("Поиск папочной структуры (train/test/pos/neg)...")

    def find_dir_case_insensitive(parent: Path, name: str) -> Path:
        for p in parent.iterdir():
            if p.is_dir() and p.name.lower() == name.lower(): return p
        for p in parent.rglob(name):
            if p.is_dir(): return p
        return None

    def parse_split(split_name: str):
        texts, labels = [], []
        split_dir = find_dir_case_insensitive(root, split_name)
        if not split_dir:
            raise FileNotFoundError(f" Папка '{split_name}' не найдена в {root}.")
        for lbl_name, lbl_id in [("pos", 1), ("neg", 0), ("positive", 1), ("negative", 0)]:
            lbl_dir = find_dir_case_insensitive(split_dir, lbl_name)
            if not lbl_dir: continue
            for txt_file in lbl_dir.glob("*.txt"):
                try:
                    texts.append(txt_file.read_text(encoding="utf-8", errors="ignore"))
                    labels.append(lbl_id)
                except Exception:
                    continue
        print(f"   {split_name}: {len(texts)} примеров")
        return Dataset.from_dict({"text": texts, "label": labels})

    return DatasetDict({"train": parse_split("train"), "test": parse_split("test")})


def load_and_preprocess_data():
    dataset = load_local_imdb_from_zip(LOCAL_ZIP_PATH, EXTRACT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    #необходимо для батчевой обработки на GPU/CPU без динамического изменения размеров тензоров.

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=4)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    train_ds = tokenized["train"].shuffle(seed=SEED)
    eval_ds = tokenized["test"].shuffle(seed=SEED)

    if TRAIN_SUBSET: train_ds = train_ds.select(range(TRAIN_SUBSET))
    if EVAL_SUBSET: eval_ds = eval_ds.select(range(EVAL_SUBSET))

    print(f" Train size: {len(train_ds)} | Eval size: {len(eval_ds)}")
    return train_ds, eval_ds, tokenizer


def compute_metrics(pred):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = np.argmax(logits, axis=1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = np.mean(preds == labels)
    return {"accuracy": float(acc), "f1": float(f1), "precision": float(precision), "recall": float(recall)}


def get_fresh_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)


def print_convergence(log_history, method_name):
    print(f"\n Сходимость ({method_name}):")
    for log in log_history:
        if "loss" in log and "eval_loss" not in log:
            print(f"  Epoch {log.get('epoch', 0):.1f} | Train Loss: {log['loss']:.4f}")
        elif "eval_loss" in log:
            print(
                f"  Epoch {log.get('epoch', 0):.1f} | Eval Loss: {log['eval_loss']:.4f} | Eval F1: {log.get('eval_f1', 0):.4f}")


def run_experiment(method_name, model, train_ds, eval_ds, lr, epochs, freeze_base=False, apply_lora=False):
    print(f"\n{'=' * 20} {method_name} {'=' * 20}")
    if freeze_base:
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name
        print(" Base model frozen. Training classifier head only.")
    if apply_lora:
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
                                 target_modules=["query", "value"], bias="none")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=f"./results_{method_name.replace(' ', '_').lower()}",
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=SEED,
        logging_steps=50,
        fp16=use_fp16,  #  Автоматически: True на GPU, False на CPU
        dataloader_num_workers=0,
        remove_unused_columns=False
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      compute_metrics=compute_metrics)
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    eval_results = trainer.evaluate()
    print_convergence(trainer.state.log_history, method_name)
    del trainer, model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return eval_results, train_time, epochs


def main():
    fix_random_state(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}")

    train_ds, eval_ds, tokenizer = load_and_preprocess_data()
    results = {}

    print("\n" + "=" * 20 + " AS-IS EVALUATION " + "=" * 20)
    model_as_is = get_fresh_model().to(device)

    use_fp16_eval = torch.cuda.is_available()

    args_as_is = TrainingArguments(
        output_dir="./results_as_is",
        per_device_eval_batch_size=32,
        report_to="none",
        seed=SEED,
        fp16=use_fp16_eval
    )
    trainer_as_is = Trainer(model=model_as_is, args=args_as_is, eval_dataset=eval_ds, compute_metrics=compute_metrics)
    start = time.time()
    eval_as_is = trainer_as_is.evaluate()
    results["As-Is"] = {"metrics": eval_as_is, "time": time.time() - start, "epochs": 0}
    print(f" As-Is Metrics: {eval_as_is}")
    del model_as_is, trainer_as_is
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    res, t, ep = run_experiment("Full FT", get_fresh_model(), train_ds, eval_ds, lr=2e-5, epochs=3)
    #lr=2e-5 (Full), 1e-4 (LP), 3e-4 (LoRA) - стандартные значения из литературы (Devlin et al., Hu et al.)
     #batch=16 - стабильный градиент при умеренном потреблении памяти
    results["Full FT"] = {"metrics": res, "time": t, "epochs": ep}

    res, t, ep = run_experiment("Linear Probing", get_fresh_model(), train_ds, eval_ds, lr=1e-4, epochs=5,
                                freeze_base=True)
    results["Linear Probing"] = {"metrics": res, "time": t, "epochs": ep}

    res, t, ep = run_experiment("LoRA", get_fresh_model(), train_ds, eval_ds, lr=3e-4, epochs=3, apply_lora=True)
    results["LoRA"] = {"metrics": res, "time": t, "epochs": ep}


    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print(f"{'Метод':<18} | {'Accuracy':<10} | {'F1':<10} | {'Время (с)':<10} | {'Эпохи'}")
    print("-" * 70)
    for method, data in results.items():
        acc = data["metrics"].get("eval_accuracy", 0)
        f1 = data["metrics"].get("eval_f1", 0)
        t = data["time"]
        ep = data["epochs"]
        print(f"{method:<18} | {acc:<10.4f} | {f1:<10.4f} | {t:<10.2f} | {ep}")
    print("=" * 70)
    print(" Эксперимент завершён. Артефакты в ./results_* | Датасет в ./data/imdb_extracted")


if __name__ == "__main__":
    main()