import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 設定定数
MODEL_NAME = "facebook/opt-125m"

# LoRAアダプターのディレクトリは、先ほど保存された場所を指定
# 実行ディレクトリ直下の 'bin/opt-125m-finetuned-lora' を想定
LORA_ADAPTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bin", "opt-125m-finetuned-lora"
)


def generate_text(prompt: str, max_new_tokens: int = 50):
    """
    ベースモデルにLoRAアダプターをロードし、プロンプトに基づいて文章を生成する
    """

    # 1. デバイスの決定 (GPUが利用可能ならcudaを使用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {device}")

    # 2. ベースモデルとトークナイザーのロード
    print(f"Loading base model: {MODEL_NAME}...")

    # RTX 40/30系向けにbfloat16を使用（get_device_capability >= 8 の確認は省略）
    # 互換性のため、GPUの場合はbf16/fp16を使用
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="auto"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. LoRAアダプターのロードとマージ
    # 保存されたLoRAアダプターのパスが存在するかチェック
    if not os.path.exists(LORA_ADAPTER_PATH):
        print(f"Error: LoRA adapter not found at path: {LORA_ADAPTER_PATH}")
        print(
            "Please ensure the fine-tuning script has run successfully and the path is correct."
        )
        return

    print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)

    # 推論パフォーマンス向上のため、アダプターをベースモデルに統合（マージ）
    # LoRAモデルを直接使用することも可能だが、推論時はマージが推奨されることが多い
    model = model.merge_and_unload()
    model.eval()  # 推論モードに設定

    # 4. 文章生成
    print("\n--- Generating Text ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # 確率的なサンプリングを有効化
            top_p=0.9,  # Top-Pサンプリング
            temperature=0.7,  # 生成のランダム性
        )

    # 5. 結果のデコードと表示
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"\nGenerated Text:\n{generated_text}")
    print("---------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a fine-tuned LoRA model."
    )
    parser.add_argument(
        "prompt", type=str, help="The input prompt to start the text generation."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="The maximum number of new tokens to generate (default: 50).",
    )

    args = parser.parse_args()
    generate_text(args.prompt, args.max_tokens)
