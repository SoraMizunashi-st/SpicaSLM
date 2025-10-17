import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# --- 1. 定数とパラメータ設定 ---
# (MITライセンスのモデル)
MODEL_NAME = "facebook/opt-125m"
# (公開されている指示応答データセット)
DATASET_NAME = "tatsu-lab/alpaca"
OUTPUT_DIR = os.path.join(
    ".", "bin", "opt-125m-finetuned-lora"
)  # binフォルダ配下に出力

# LoRAのパラメータ
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# トレーニングパラメータ
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-4

# --- 2. データセットの前処理関数 ---


def format_instruction(example):
    """
    データセットのフォーマットをInstruction Tuning形式に変換する関数
    (OPTモデルが理解できるように、入力と出力を結合する)
    """
    # Alpacaデータセットの標準的なフォーマットを使用
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n"
    # 入力があればそれも含める
    if example.get("input"):
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"

    # モデルの学習に必要なのは入力+出力のペア
    return {"text": prompt + example["output"]}


# --- 3. メイン処理 ---


def main():
    # GPUの確認と初期化
    accelerator = Accelerator()
    device = accelerator.device

    print(f"Loading model: {MODEL_NAME} on device: {device}")

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # トークナイザーにパディングトークンを追加（LoRAとバッチ処理で必須）
    tokenizer.pad_token = tokenizer.eos_token

    # モデルのロード (bf16で省メモリ化、Q-LoRAの準備)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # CUDAが利用可能かつCompute Capability >= 8 (Ampere世代以降) ならbf16、そうでなければfp16を使用
        torch_dtype=torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16,
        device_map="auto",
    )

    # 勾配チェックポイント処理（メモリ削減）を有効化
    model.gradient_checkpointing_enable()
    # LoRAでトレーニング可能にするためにモデルを準備
    model = prepare_model_for_kbit_training(model)

    # LoRA設定の定義と適用
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # データセットのロード
    print(f"Loading dataset: {DATASET_NAME}")
    # Alpacaデータセットの小さなサブセットを使用（チュートリアル用、500件）
    dataset = load_dataset(DATASET_NAME, split="train[:500]")

    # データセットをフォーマットしてトークナイズ
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )

    tokenized_datasets = dataset.map(format_instruction).map(
        tokenize_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )

    # ------------------------------------------------
    # ↓↓↓ 【重要】 追加するコード（ラベルの作成） ↓↓↓
    # ------------------------------------------------
    def create_labels(examples):
        # 因果言語モデリング（CLM）では、input_ids自体が正解ラベルとなります
        examples["labels"] = examples["input_ids"]
        return examples

    tokenized_datasets = tokenized_datasets.map(create_labels, batched=True)
    # ------------------------------------------------
    # ↑↑↑ 【重要】 追加するコード（ラベルの作成） ↑↑↑
    # ------------------------------------------------

    # 訓練用と評価用に分割
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

    # トレーニング引数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        save_strategy="epoch",
        overwrite_output_dir=True,
        fp16=True,  # 混合精度学習を有効化
    )

    # Trainerの定義と学習の実行
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )

    print("Starting fine-tuning...")
    trainer.train()

    # LoRAアダプターのみを保存（バイナリをbinに出力）
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuning completed. LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
