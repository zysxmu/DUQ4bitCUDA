import argparse
import random
from utils import *
from quant import *
import pickle
import time
import rich
from rich.progress import track
from rich.panel import Panel

from lib.qlinear4bit.nn import LinearQuant4bitDUASQ, Linear4bitASQ, LinearQuant8bitASQ

def get_args_parser():
    parser = argparse.ArgumentParser(description="ERQ-ViT", add_help=False)
    parser.add_argument("--model", default="vit_base",
                        choices=['vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base',
                                 'swin_tiny', 'swin_small', 'swin_base'],
                        help="model")
    parser.add_argument('--use-fake-data', default=True, type=bool,
                        help='simply use a fake dataset to benchmark the performance')
    parser.add_argument('--fake-data-len', default=2000, type=int,
                        help='length of the fake dataset')
    parser.add_argument('--fake-num-classes', default=1000, type=int,
                        help='number of classes of the fake dataset')
    parser.add_argument('--fake-image-size', default=(3, 224, 224), type=int,
                        help='image size')
    parser.add_argument('--dataset', default="/dataset/imagenet/",
                        help='path to dataset')
    parser.add_argument("--calib-batchsize", default=1024,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--val-batchsize", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=16, type=int,
                        help="number of data loading workers (default: 16)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--print-freq", default=100,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    parser.add_argument('--w_bits', default=4,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=4,
                        type=int, help='bit-precision of activation')
    parser.add_argument('--coe', default=20000,
                        type=int, help='')

    return parser

model_zoo = {
    'vit_small': 'vit_small_patch16_224',
    'vit_base': 'vit_base_patch16_224',
    'vit_large': 'vit_large_patch16_224',
    
    'deit_tiny': 'deit_tiny_patch16_224',
    'deit_small': 'deit_small_patch16_224',
    'deit_base': 'deit_base_patch16_224',

    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base': 'swin_base_patch4_window7_224'
}

def quant_model_cuda_4bit(model, input_quant_params={}, weight_quant_params={}, use_duq=True):
    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2['log_quant'] = True

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True
    
    rich.print(rich.panel.Panel("[bold red]Start quantizing model with \[DUQLinear4bitCUDA]...[/bold red]", 
                                expand=False, title="Quantization Tool"))

    module_dict={}
    for name, m in rich.progress.track(model.named_modules(), description="Quantizing model"):
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            # if 'qkv' in name or 'fc1' in name or 'reduction' in name:
            if 'qkv' in name or 'fc1' in name:
                if use_duq:
                    new_m = LinearQuant4bitDUASQ.from_linear(m, require_quantizer=True)
                else:
                    new_m = Linear4bitASQ.from_linear(m, require_quantizer=True)
                rich.print(f"[blue]Find [green bold]QKV-Linear[/green bold]: [yellow]{name}[/yellow], "
                    "using [yellow]'LinearQuant4bit([red]DualUniformAsymQuant[/red])'[/yellow] to quantize...[/blue]")
            else:
                new_m = Linear4bitASQ.from_linear(m, require_quantizer=True)
                rich.print(f"[blue]Find [green bold]QKV-Linear[/green bold]: [yellow]{name}[/yellow], "
                    "using [yellow]'LinearQuant4bit([red]UniformAsymQuant[/red])'[/yellow] to quantize...[/blue]")
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
    return model

def quant_model_cuda_8bit(model, input_quant_params={}, weight_quant_params={}):
    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2['log_quant'] = True

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True
    
    rich.print(rich.panel.Panel("[bold red]Start quantizing model with \[DUQLinear4bitCUDA]...[/bold red]", 
                                expand=False, title="Quantization Tool"))

    module_dict={}
    for name, m in rich.progress.track(model.named_modules(), description="Quantizing model"):
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            # if 'qkv' in name or 'fc1' in name or 'reduction' in name:
            new_m = LinearQuant8bitASQ.from_linear(m, require_quantizer=True)
            rich.print(f"[blue]Find [green bold]QKV-Linear[/green bold]: [yellow]{name}[/yellow], "
                "using [yellow]'LinearQuant8bit([red]AsymQuant[/red])'[/yellow] to quantize...[/blue]")
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
    return model

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def benchmark_model(model, val_loader, device, dtype=torch.float16):
    rich.print(rich.panel.Panel(f"Start benchmarking model {model.__class__.__name__}", title="Benchmarking", expand=False))
    model.to(device)
    model.eval()
    
    rich.print('[red bold]Running warm-up...[/red bold]')
    for data_batch in track(val_loader):
        input = data_batch[0].to(device).to(dtype)
        with torch.no_grad():
            model(input)
    torch.cuda.synchronize()
    
    rich.print('[green]Warmup done.[green]')
    rich.print('[blue bold]Running Benchmark...[/blue bold]')
    
    start_time = time.perf_counter()
    for data_batch in val_loader:
        input = data_batch[0].to(device).to(dtype)
        with torch.no_grad():
            model(input)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = (end_time - start_time) * 1000
    avg_batch_time = total_time / len(val_loader)
    
    rich.print('[blue green]Done.[/blue green]')
    
    return total_time, avg_batch_time

def show_results(name, total_time, avg_batch_time):
    rich.print(Panel(f"[yellow bold]ViT Benchmark - {name}(bs={args.val_batchsize}).[/yellow bold]\n[blue]Average Inference Time: [/blue][green bold]{total_time:.3f} ms[/green bold]\n[magenta]Avergae batch inference time: [/magenta][green bold]{avg_batch_time:.3f} ms[/green bold]", title="SpeedUP", expand=False))

def main():
    print(args)
    seed(args.seed)
    
    device = torch.device(args.device)
    
    print("Building dataloader...")
    val_loader = build_randval_dataset(args)
    
    print("Building model...")
    model = build_model(model_zoo[args.model])
    model.to(device)
    model.eval()
    
    fp32_tot, fp32_avg = benchmark_model(model, val_loader, device, dtype=torch.float32)
    
    del(model)
    
    model = build_model(model_zoo[args.model])
    model.to(device)
    model.eval()
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model_cuda_4bit(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device).to(torch.float16)
    q_model.eval()
    
    int4duq_tot, int4duq_avg = benchmark_model(q_model, val_loader, device, dtype=torch.float16)
    int4duq_tot, int4duq_avg = benchmark_model(q_model, val_loader, device, dtype=torch.float16)
    
    del(model); del(q_model)
    
    model = build_model(model_zoo[args.model])
    model.to(device)
    model.eval()
    model.to(torch.float16)
    
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model_cuda_4bit(model, input_quant_params=aq_params, weight_quant_params=wq_params, use_duq=False)
    q_model.to(device)
    q_model.eval()
    
    int4uq_tot, int4uq_avg = benchmark_model(q_model, val_loader, device, dtype=torch.float16)
    
    del(model); del(q_model)
    
    
    model = build_model(model_zoo[args.model])
    model.eval()
    
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model_cuda_8bit(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()
    
    int8_tot, int8_avg = benchmark_model(q_model, val_loader, device, dtype=torch.float32)
    
    show_results(f'{args.model}-FP32', fp32_tot, fp32_avg)
    show_results(f'{args.model}-Int8', int8_tot, int8_avg)
    show_results(f'{args.model}-Int4(UQ)', int4uq_tot, int4uq_avg)
    show_results(f'{args.model}-Int4(DUQ)', int4duq_tot, int4duq_avg)
    
    rich.print(rich.panel.Panel(f"FP32 -> INT8: {fp32_avg / int8_avg:.4f}x\nINT8 -> INT4(UQ): {int8_avg / int4uq_avg:.4f}x\nFP32 -> INT4(UQ): {fp32_avg / int4uq_avg:.4f}x\nINT8 -> INT4(DUQ): {int8_avg / int4duq_avg:.4f}x\nFP32 -> INT4(DUQ): {fp32_avg / int4duq_avg:.4f}x", title="Speedup", expand=False))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('ERQ', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
