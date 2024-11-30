import torch
from lib.qlinear4bit.nn import LinearQuant4bitDUASQ, Linear4bitASQ, LinearQuant8bitASQ
import time
import rich
from rich.table import Table
import argparse
import numpy as np
import pprint

model_sizes = [
    (4096, 4096), #llama-7b
    (5120, 5120), #llama-13b
    (8192, 8192)  #llama-70b   
]

mlp_sizes = [
    (4096, 11008), #llama-7b
    (5120, 13824), #llama-13b
    (8192, 28672)  #llama-70b
]
benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 100


def module_benchmark(module, x):
    x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000 / num_bench_steps

def show_tables(title, data_8b, data_4b, data_4b2, data_baseline):
    table = Table(title=title + f'(type={args.layer_type}, seqlen={args.seq_len}, batchsize={args.bsz})')
    table.add_column("module type", justify="center", style="magenta", no_wrap=True)
    for key in data_8b.keys():
        table.add_column(key, justify="center", style="cyan", no_wrap=True)
    table.add_row("LinearFP32", *[data_baseline[key] for key in data_baseline.keys()])
    table.add_row("LinearInt8(with quantizer)", *[data_8b[key] for key in data_8b.keys()])
    table.add_row("LinearInt4(UniformQuant, with quantizer)", *[data_4b2[key] for key in data_4b2.keys()])
    table.add_row("LinearInt4(DualUniformQuant, with quantizer)", *[data_4b[key] for key in data_4b.keys()])
    table.add_section()
    table.add_row("[green]Speedup\[FP32->INT8][/green]", *[f'{float(data_baseline[key].split()[0]) / float(data_8b[key].split()[0]):.3f}x' for key in data_8b.keys()])
    table.add_row("[green]Speedup\[FP32->INT4(DUQ)][/green]", *[f'{float(data_baseline[key].split()[0]) / float(data_4b[key].split()[0]):.3f}x' for key in data_4b.keys()])
    table.add_row("[green]Speedup\[INT8->INT4(DUQ)][/green]", *[f'{float(data_8b[key].split()[0]) / float(data_4b[key].split()[0]):.3f}x' for key in data_4b.keys()])
    table.add_row("[blue]Speedup\[FP32->INT4(UQ)][/blue]", *[f'{float(data_baseline[key].split()[0]) / float(data_4b2[key].split()[0]):.3f}x' for key in data_4b.keys()])
    table.add_row("[blue]Speedup\[INT8->INT4(UQ)][/blue]", *[f'{float(data_8b[key].split()[0]) / float(data_4b2[key].split()[0]):.3f}x' for key in data_4b.keys()])
    
    rich.print(table)
    
def linear4bit_benchmark(args):
    bsz = args.bsz
    seq_len = args.seq_len
    
    if args.layer_type == 'v_proj':
        layer_size = model_sizes
    else:
        layer_size = mlp_sizes
        
    data_8b = dict()
    data_4b = dict()
    data_4b2 = dict()
    data_baseline = dict()
    
    for (feature_dim_in, feature_dim_out) in layer_size:
        for dtype in benchmark_dtypes:
            
            x = torch.rand((bsz,
                            seq_len,
                            feature_dim_in)).cuda().to(dtype)
            
            baseline_mod = torch.nn.Linear(feature_dim_in,
                                           feature_dim_out,
                                           bias=False).cuda().to(dtype)
            
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data,
                                                          low=-8, high=7).to(dtype)
            
            int8asq_mod = LinearQuant8bitASQ.from_linear(baseline_mod).cuda()
            
            times_8bit = []
            for i in range(10):
                times_8bit.append(module_benchmark(int8asq_mod, x.to(torch.float32)))
            print(f"Int8-ASQ time: {np.mean(times_8bit):.3f} +- {1.96 * np.std(times_8bit):.3f}ms")
            
            data_8b[f'DIM_IN:{feature_dim_in}|DIM_OUT:{feature_dim_out}'] = f'{np.mean(times_8bit):.3f} +- {1.96 * np.std(times_8bit):.3f}ms'
            
            int4duasq_mod = LinearQuant4bitDUASQ.from_linear(baseline_mod).cuda()

            times_4bitduq = []
            for i in range(10):
                times_4bitduq.append(module_benchmark(int4duasq_mod, x))
            print(f"Int4-DualUniformQuant time: {np.mean(times_4bitduq):.3f} +- {1.96 * np.std(times_4bitduq):.3f}ms")
            
            data_4b[f'DIM_IN:{feature_dim_in}|DIM_OUT:{feature_dim_out}'] = f'{np.mean(times_4bitduq):.3f} +- {1.96 * np.std(times_4bitduq):.3f}ms'
            
            int4asq_mod = Linear4bitASQ.from_linear(baseline_mod).cuda()
            times_4bitasq = []
            for i in range(10):
                times_4bitasq.append(module_benchmark(int4asq_mod, x))
            print(f"Int4-ASQ time: {np.mean(times_4bitasq):.3f} +- {1.96 * np.std(times_4bitasq):.3f}ms")
            data_4b2[f'DIM_IN:{feature_dim_in}|DIM_OUT:{feature_dim_out}'] = f'{np.mean(times_4bitasq):.3f} +- {1.96 * np.std(times_4bitasq):.3f}ms'
                        
            times_baseline = []
            for i in range(10):
                times_baseline.append(module_benchmark(baseline_mod, x))
            print(f"FP32 time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
            
            data_baseline[f'DIM_IN:{feature_dim_in}|DIM_OUT:{feature_dim_out}'] = f'{np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms'

            print(f"Speedup(baeline -> Int4-DUQ): {np.mean(times_baseline) / np.mean(times_4bitduq):.3f}x")
            print(f"Speedup(baeline -> Int8-ASQ): {np.mean(times_baseline) / np.mean(times_8bit):.3f}x")
            print(f"Speedup(Int8-ASQ -> Int4-DUQ): {np.mean(times_8bit) / np.mean(times_4bitduq):.3f}x")
            print(f"Speedup(Int8-ASQ -> Int4-ASQ): {np.mean(times_8bit) / np.mean(times_4bitasq):.3f}x")
            
            # table-style output
            print(f'{feature_dim_in}x{feature_dim_out} & {args.bsz} & {np.mean(times_baseline):.3f}\\\\')
            print('--------------')
    
    show_tables("Linear Module Benchmark", data_8b, data_4b, data_4b2, data_baseline)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--layer_type', type=str,
        help='Type of the layer in the model (v_proj [default], down_proj)',
        default='v_proj',
        choices=['v_proj', 'down_proj']
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    linear4bit_benchmark(args)
