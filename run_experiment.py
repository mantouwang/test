import subprocess
import sys
import os

def run_fusion_model_experiment():
    """
    这是一个Python启动器脚本，用于配置参数并运行主训练程序 a_transductive.py。
    """
    print("=" * 60)
    print("Starting Fusion Model Training Experiment...")
    print("=" * 60)

    # 使用sys.executable可以确保我们调用的是当前conda环境中的python解释器，这是一种更稳健的做法
    python_executable = sys.executable

    # 将所有命令行参数构建成一个列表
    # 每个参数和它的值都作为列表中的独立字符串
    command = [
        python_executable,
        "main_transductive.py",
        "--device", "0",
        "--seeds", "42",
        "--max_epoch", "1500",
        "--lr", "0.001",
        "--num_hidden", "512",
        "--encoder", "gat",
        "--decoder", "gat",
        "--num_heads", "4",
        "--weight_decay", "0.0001",
        "--activation", "relu",
        "--in_drop", "0.2",
        "--attn_drop", "0.1",
        "--norm", "batchnorm",
        "--residual",
        "--scheduler",
        "--cache_path", "E:/下载/SMG-main/aligned_data.pt",
        "--vae_latent_dim", "16",
        "--fusion_hidden_dim", "256",
        "--fusion_out_dim", "512",
    ]

    # 为了方便调试，打印出将要执行的完整命令
    print("Executing command:")
    # os.linesep是换行符，让长命令更易读
    print(" ".join(command).replace("--", f"--{os.linesep}\t"))
    print("-" * 60)

    try:
        # 执行命令。subprocess.run会等待命令执行完成
        # check=True 表示如果子进程返回非零退出码（即发生错误），则会抛出异常
        subprocess.run(command, check=True)
        print("\n" + "=" * 60)
        print("Training script finished successfully!")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print(f"ERROR: Training script failed with exit code {e.returncode}.")
        print("Please check the error messages above.")
        print("=" * 60)
    except FileNotFoundError:
        print("\n" + "=" * 60)
        print(f"ERROR: Could not find '{python_executable}' or 'main_transductive.py'.")
        print("Please ensure this script is in the same directory as main_transductive.py.")
        print("=" * 60)

if __name__ == "__main__":
    run_fusion_model_experiment()