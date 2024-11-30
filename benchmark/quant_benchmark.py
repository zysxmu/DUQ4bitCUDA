import time
import rich
import rich.progress
import torch
import numpy as np
from rich import print
import lib.qlinear4bit.tools.tensor_utils as qcu_tool
from lib.qlinear4bit.nn import AsymQuantizer, AsymQuantizer8bit

m_sizes = [
    (4096, 4096), 
    (5120, 5120), 
    (8192, 8192)     
]
SEQLEN = 2048

benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 100

def dequant_benchamrk(shape):
    matrix = torch.randint(0, 127, shape).cuda().to(torch.int32)
    fake_row_scales = torch.rand(shape[0]).cuda()
    fake_row_zeros = torch.rand(shape[0]).cuda()
    fake_col_scales = torch.rand(shape[1]).cuda()
    fake_col_zeros = torch.rand(shape[1]).cuda()
    
    time_dequant_8bit = []
    for i in range(10):
        for _ in rich.progress.track(range(num_warmup_steps), description="Running warmup for 8bit dequant"):
            _out = qcu_tool.asym_dequant_hprec(matrix, fake_row_scales, fake_row_zeros, fake_col_scales, fake_col_zeros)
        print("[bold red]Start benchmarking({}/{})...[bold /red]".format(i+1, 10))
        for _ in range(num_bench_steps):
            start_time = time.perf_counter()
            _out = qcu_tool.asym_dequant_hprec(matrix, fake_row_scales, fake_row_zeros, fake_col_scales, fake_col_zeros)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_dequant_8bit.append((end_time - start_time) * 1000 / num_bench_steps)
        print("[bold green]End benchmarking({}/{})...[bold /green]".format(i+1, 10))
    print(f"8bit dequant time: {np.mean(time_dequant_8bit)} ms")
    
    fake_row_scales = fake_row_scales.to(torch.float16)
    fake_row_zeros = fake_row_zeros.to(torch.float16)
    fake_col_scales = fake_col_scales.to(torch.float16)
    fake_col_zeros = fake_col_zeros.to(torch.float16)
    
    time_dequant_4bit = []
    for i in range(10):
        for _ in rich.progress.track(range(num_warmup_steps), description='Running warmup for 4bit dequant'):
            _out = qcu_tool.asym_dequant(matrix, fake_row_scales, fake_row_zeros, fake_col_scales, fake_col_zeros)
        print("[bold red]Start benchmarking({}/{})...[bold /red]".format(i+1, 10))
        for _ in range(num_bench_steps):
            start_time = time.perf_counter()
            _out = qcu_tool.asym_dequant(matrix, fake_row_scales, fake_row_zeros, fake_col_scales, fake_col_zeros)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_dequant_4bit.append((end_time - start_time) * 1000 / num_bench_steps)
        print("[bold green]End benchmarking({}/{})...[bold /green]".format(i+1, 10))
    print(f"4bit dequant time: {np.mean(time_dequant_4bit)} ms")
    return np.mean(time_dequant_8bit), np.mean(time_dequant_4bit)


def quant_benchmark(shapeA, shapeB):
    matrixA = torch.randint(0, 127, shapeA).cuda().to(torch.float32)
    matrixB = torch.randint(0, 127, shapeB).cuda().to(torch.float32)
    
    quantizer_a = AsymQuantizer8bit()
    # quantizer_b = AsymQuantizer8bit()
    
    time_mm8bit = []
    for i in range(10):
        for _ in rich.progress.track(range(num_warmup_steps), description="Running warmup for 8bit quant"):
            _out = quantizer_a(matrixA)
        print("[bold red]Start benchmarking({}/{})...[bold /red]".format(i+1, 10))
        for _ in range(num_bench_steps):
            start_time = time.perf_counter()
            _out  = quantizer_a(matrixA)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_mm8bit.append((end_time - start_time) * 1000 / num_bench_steps)
        print("[bold green]End benchmarking({}/{})...[bold /green]".format(i+1, 10))
        # time_mm8bit.append(benmark_func(qcu_tool.matmul_8bit, matrixA_8b, matrixB_8b))
    print(f"8bit quant time: {sum(time_mm8bit)/len(time_mm8bit)} ms")
    
    del(quantizer_a)
    
    matrixA = matrixA.to(torch.float16)
    matrixB = matrixB.to(torch.float16)
    
    quantizer_a4b = AsymQuantizer()
    
    time_mm4bit = []
    for i in range(10):
        for _ in rich.progress.track(range(num_warmup_steps), description="Running warmup for 4bit quant"):
            _out = quantizer_a4b(matrixA)
        print("[bold red]Start benchmarking({}/{})...[bold /red]".format(i+1, 10))
        for _ in range(num_bench_steps):
            start_time = time.perf_counter()
            _out = quantizer_a4b(matrixA)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_mm4bit.append((end_time - start_time) * 1000 / num_bench_steps)
        print("[bold green]End benchmarking({}/{})...[bold /green]".format(i+1, 10))
        # time_mm4bit.append(benmark_func(qcu_tool.matmul, matrixA_4b, matrixB_4b))
    print(f"4bit quant time: {sum(time_mm4bit)/len(time_mm4bit)} ms")
    return np.mean(time_mm8bit), np.mean(time_mm4bit)

def matmul_benchmark(shapeA, shapeB):
    matrixA = torch.randint(0, 127, shapeA).cuda()
    matrixB = torch.randint(0, 127, shapeB).cuda()
    
    matrixA_8b = matrixA.to(torch.int8)
    matrixB_8b = matrixB.to(torch.int8)
    
    time_mm8bit = []
    for i in range(10):
        for _ in rich.progress.track(range(num_warmup_steps), description="Running warmup for 8bit matmul"):
            out = qcu_tool.matmul_8bit(matrixA_8b, matrixB_8b)
        print("[bold red]Start benchmarking({}/{})...[bold /red]".format(i+1, 10))
        for _ in range(num_bench_steps):
            start_time = time.perf_counter()
            out = qcu_tool.matmul_8bit(matrixA_8b, matrixB_8b)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_mm8bit.append((end_time - start_time) * 1000 / num_bench_steps)
        print("[bold green]End benchmarking({}/{})...[bold /green]".format(i+1, 10))
    print(f"8bit matmul time: {sum(time_mm8bit)/len(time_mm8bit)} ms")
    
    del(matrixA_8b)
    del(matrixB_8b)
    
    matrixA = matrixA.to(torch.float16)
    matrixB = matrixB.to(torch.float16)
    
    matrixA_4b = AsymQuantizer()(matrixA).quantized_x
    matrixB_4b = AsymQuantizer()(matrixB).quantized_x
    
    time_mm4bit = []
    for i in range(10):
        for _ in rich.progress.track(range(num_warmup_steps), description="Running warmup for 4bit matmul"):
            out = qcu_tool.matmul(matrixA_4b, matrixB_4b)
        print("[bold red]Start benchmarking({}/{})...[bold /red]".format(i+1, 10))
        for _ in range(num_bench_steps):
            start_time = time.perf_counter()
            out = qcu_tool.matmul(matrixA_4b, matrixB_4b)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_mm4bit.append((end_time - start_time) * 1000 / num_bench_steps)
        print("[bold green]End benchmarking({}/{})...[bold /green]".format(i+1, 10))
    print(f"4bit matmul time: {sum(time_mm4bit)/len(time_mm4bit)} ms")
    return np.mean(time_mm8bit), np.mean(time_mm4bit)
    

def pipeline_benchmark(shapeA, shapeB):
    matrixA = torch.randint(0, 127, shapeA).cuda().to(torch.float32)
    matrixB = torch.randint(0, 127, shapeB).cuda().to(torch.float32)
    
    matrixB_Q8bit = AsymQuantizer8bit()(matrixB)
    
    quantizer_8bit = AsymQuantizer8bit()
    
    time_pp8bit = []
    for i in range(10):
        for _ in rich.progress.track(range(num_warmup_steps), description="Running warmup for 8bit pipeline"):
            matrixA_8b = quantizer_8bit(matrixA)
            out = qcu_tool.matmul_8bit(matrixA_8b.quantized_x, matrixB_Q8bit.quantized_x)
            dequant_out = qcu_tool.asym_dequant_hprec(out, matrixA_8b.scales_x, matrixA_8b.zeros_x, matrixB_Q8bit.scales_x, matrixB_Q8bit.zeros_x)
        print("[bold red]Start benchmarking({}/{})...[bold /red]".format(i+1, 10))
        for _ in range(num_bench_steps):
            start_time = time.perf_counter()
            matrixA_8b = quantizer_8bit(matrixA)
            out = qcu_tool.matmul_8bit(matrixA_8b.quantized_x, matrixB_Q8bit.quantized_x)
            dequant_out = qcu_tool.asym_dequant_hprec(out, matrixA_8b.scales_x, matrixA_8b.zeros_x, matrixB_Q8bit.scales_x, matrixB_Q8bit.zeros_x)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_pp8bit.append((end_time - start_time) * 1000 / num_bench_steps)
        print("[bold green]End benchmarking({}/{})...[bold /green]".format(i+1, 10))
    print(f"8bit pipeline time: {sum(time_pp8bit)/len(time_pp8bit)} ms")
    
    del(quantizer_8bit)
    
    matrixA = matrixA.to(torch.float16)
    matrixB = matrixB.to(torch.float16)
    
    matrixB_Q4bit = AsymQuantizer()(matrixB)
    
    quantizer_4bit = AsymQuantizer()
    
    time_pp4bit = []
    for i in range(10):
        for _ in rich.progress.track(range(num_warmup_steps), description="Running warmup for 4bit pipeline"):
            matrixA_4b = quantizer_4bit(matrixA)
            out = qcu_tool.matmul(matrixA_4b.quantized_x, matrixB_Q4bit.quantized_x)
            dequant_out = qcu_tool.asym_dequant(out, matrixA_4b.scales_x, matrixA_4b.zeros_x, matrixB_Q4bit.scales_x, matrixB_Q4bit.zeros_x)
        print("[bold red]Start benchmarking({}/{})...[bold /red]".format(i+1, 10))
        for _ in range(num_bench_steps):
            start_time = time.perf_counter()
            matrixA_4b = quantizer_4bit(matrixA)
            out = qcu_tool.matmul(matrixA_4b.quantized_x, matrixB_Q4bit.quantized_x)
            dequant_out = qcu_tool.asym_dequant(out, matrixA_4b.scales_x, matrixA_4b.zeros_x, matrixB_Q4bit.scales_x, matrixB_Q4bit.zeros_x)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_pp4bit.append((end_time - start_time) * 1000 / num_bench_steps)
        print("[bold green]End benchmarking({}/{})...[bold /green]".format(i+1, 10))
    print(f"4bit pipeline time: {sum(time_pp4bit)/len(time_pp4bit)} ms")
    return np.mean(time_pp8bit), np.mean(time_pp4bit)
    
    
def show_results(name, total_time, avg_batch_time):
    print(rich.panel.Panel(f"8bit Avg Time: {total_time} ms; \n4bit Avg Time: {avg_batch_time} ms", expand=False, title=f"[bold red]{name}[/bold red] - Results"))

def show_tables(title, times_8b, times_4b):
    table = rich.table.Table(title=title)
    table.add_column("bit type", justify="center", style="magenta", no_wrap=True)
    for key in times_8b.keys():
        table.add_column(str(key), justify="center", style="cyan", no_wrap=True)
    table.add_row("8bit", *[f"{time:.6f} ms" for time in times_8b.values()])
    table.add_row("4bit", *[f"{time:.6f} ms" for time in times_4b.values()])
    table.add_section()
    table.add_row("[bold green]SpeedUp[/bold green]", *[f"[yellow]{times_8b[key]/times_4b[key]:.3f} x[/yellow]" for key in times_8b.keys()])
    print(table)

def main():
    print(rich.panel.Panel("Quant benchmark", expand=False))
    time_q8b, time_q4b = dict(), dict()
    for m_size in m_sizes:
        matrixA_size = (SEQLEN, m_size[0])
        matrixB_size = m_size
        print(f"Matrix A size: {matrixA_size}; Matrix B size: {matrixB_size}")
        time_q8b_, time_q4b_ = quant_benchmark(m_size, m_size)
        time_q8b[matrixA_size] = time_q8b_
        time_q4b[matrixA_size] = time_q4b_
    
    time_dq8b, time_dq4b = dict(), dict()
    print(rich.panel.Panel("Dequant benchmark", expand=False))
    for m_size in m_sizes:
        matrix_size = (SEQLEN, m_size[0])
        print(f"Matrix size: {matrix_size}")
        time_dq8b_, time_dq4b_ = dequant_benchamrk(matrix_size)
        time_dq8b[matrix_size] = time_dq8b_
        time_dq4b[matrix_size] = time_dq4b_
    
    time_mm8b, time_mm4b = dict(), dict()
    print(rich.panel.Panel("Matmul benchmark", expand=False))
    for m_size in m_sizes:
        matrixA_size = (SEQLEN, m_size[0])
        matrixB_size = m_size
        print(f"Matrix A size: {matrixA_size}; Matrix B size: {matrixB_size}")
        time_mm8b_, time_mm4b_ = matmul_benchmark(m_size, m_size)
        time_mm8b[f'{matrixA_size} * {m_size}'] = time_mm8b_
        time_mm4b[f'{matrixA_size} * {m_size}'] = time_mm4b_
        
    time_pp8b, time_pp4b = dict(), dict()
    print(rich.panel.Panel("Pipeline benchmark", expand=False))
    for m_size in m_sizes:
        matrixA_size = (SEQLEN, m_size[0])
        matrixB_size = m_size
        print(f"Matrix A size: {matrixA_size}; Matrix B size: {matrixB_size}")
        time_pp8b_, time_pp4b_ = pipeline_benchmark(m_size, m_size)
        time_pp8b[f'SEQ-LEN:{SEQLEN}|DIM:IN{m_size[0]}/OUT{m_size[1]}'] = time_pp8b_
        time_pp4b[f'SEQ-LEN:{SEQLEN}|DIM:IN{m_size[0]}/OUT{m_size[1]}'] = time_pp4b_
    
    show_tables("Quantization Benchmark", time_q8b, time_q4b)
    show_tables("Dequantization Benchmark", time_dq8b, time_dq4b)
    show_tables("Matmul Benchmark", time_mm8b, time_mm4b)
    show_tables("Pipeline Benchmark", time_pp8b, time_pp4b)
    
    
if __name__ == "__main__":
    main()
    
