## Cuda Implementation for Dual Uniform Quantization

This code is for evaluate the real run time of Dual Uniform Quantization proposed in 'Towards Accurate Post-Training Quantization of Vision Transformers via Error Reduction'

## Basic Idea

We implemented quantization, matrix multiplication, and dequantization kernels for 8-bit and 4-bit precision based on the `Cutlass` library (referencing the implementation of [QuaRot](https://github.com/spcl/QuaRot)) and provided the corresponding test scripts:
1. Linear module benchmark
2. Full vision transformer benchmark

## Results

1. Linear module benchmark results
   ![LinearResults](https://github.com/user-attachments/assets/988b5094-96ab-4bbd-b301-c98e50780818)
2. Full Vit model benchmark results
   ![ViTBenchmark](https://github.com/user-attachments/assets/4913a9e8-6ca1-46cc-b3b8-31eb43a82a4b)

## Environment Setup

You can run `bash env_setup.sh` to configure the environment automatically.

## Benchmark

### Linear Module Benchmark

- You can simply run above command to benchmark the performance of Linear module(for FP32, 8bit, and 4bit mode)

  ```
  export PYTHONPATH=./
  python benchmark/module_benchmark.py --bsz=<BatchSize> --seq_len=<SequenceLength> --layer_type=<LayerType>
  ```
- The `LayerType` can be `v_proj` or `down_proj`.

### Full Vit Model Benchmark

- You can simply run above command to benchmark the performance of Linear module(for FP32, 8bit, and 4bit mode)

  ```
  export PYTHONPATH=./
  python benchmark/vit_benchmark.py --model=<ModelType> --val-batchsize=<ValBatchSize>
  ```

## Acknowledge

Our code is heavily based on the cuda code of [QuaRot](https://github.com/spcl/QuaRot). We highly appreciate their work.

```bash
@article{ashkboos2024quarot,
  title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs},
  author={Ashkboos, Saleh and Mohtashami, Amirkeivan and Croci, Maximilian L and Li, Bo and Jaggi, Martin and Alistarh, Dan and Hoefler, Torsten and Hensman, James},
  journal={arXiv preprint arXiv:2404.00456},
  year={2024}
}
```
