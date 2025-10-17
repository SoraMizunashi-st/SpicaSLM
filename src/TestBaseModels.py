import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 設定定数
MODEL_NAME = "facebook/opt-125m"


def generate_text_from_base(prompt: str, max_new_tokens: int = 50):
    """
    ベースモデル（LoRAアダプターなし）をロードし、プロンプトに基づいて文章を生成する
    """

    # 1. デバイスの決定 (GPUが利用可能ならcudaを使用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {device}")

    # 2. ベースモデルとトークナイザーのロード
    print(f"Loading base model: {MODEL_NAME}...")

    # RTX 40/30系向けにbfloat16を使用 (Compute Capability >= 8)
    # CPU環境でも動作するよう、適切なdtypeを選択
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32  # CPUの場合はfloat32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="auto"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()  # 推論モードに設定

    # 3. 文章生成
    print("\n--- Generating Text from Base Model ---")
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

    # 4. 結果のデコードと表示
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"\nGenerated Text:\n{generated_text}")
    print("---------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using the base model (no fine-tuning)."
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
    generate_text_from_base(args.prompt, args.max_tokens)
