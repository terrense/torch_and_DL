#!/usr/bin/env python3
"""
Performance profiling script for Paraformer ASR.
Measures memory usage, training throughput, and inference latency.
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models.paraformer import Paraformer
from src.data.toy_seq2seq import ToySeq2SeqDataset
from src.data.tokenizer import CharTokenizer
from src.data.utils import collate_fn
from torch.utils.data import DataLoader


class PerformanceProfiler:
    def __init__(self, config_path: str, device: str = 'cuda'):
        self.config = load_config(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = CharTokenizer(vocab_size=self.config.data.vocab_size)
        
        # Initialize model
        self.model = Paraformer(
            vocab_size=self.tokenizer.vocab_size,
            feature_dim=self.config.data.feature_dim,
            **self.config.model.__dict__
        ).to(self.device)
        
        self.results = {}
        
    def profile_memory(self, batch_size: int = 16, seq_len: int = 200):
        """Profile GPU memory usage during forward and backward pass."""
        print(f"\n{'='*60}")
        print(f"PROFILING MEMORY USAGE (batch_size={batch_size}, seq_len={seq_len})")
        print(f"{'='*60}")
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory profiling")
            return
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Create dummy batch
        features = torch.randn(batch_size, seq_len, self.config.data.feature_dim).to(self.device)
        tokens = torch.randint(0, self.tokenizer.vocab_size, (batch_size, seq_len // 4)).to(self.device)
        feat_lens = torch.full((batch_size,), seq_len, dtype=torch.long).to(self.device)
        token_lens = torch.full((batch_size,), seq_len // 4, dtype=torch.long).to(self.device)
        
        # Measure memory before forward
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Forward pass
        logits = self.model(features, tokens, feat_lens, token_lens)
        torch.cuda.synchronize()
        mem_after_forward = torch.cuda.memory_allocated() / 1024**2
        
        # Backward pass
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
        loss.backward()
        torch.cuda.synchronize()
        mem_after_backward = torch.cuda.memory_allocated() / 1024**2
        
        # Peak memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        results = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'memory_before_mb': round(mem_before, 2),
            'memory_after_forward_mb': round(mem_after_forward, 2),
            'memory_after_backward_mb': round(mem_after_backward, 2),
            'peak_memory_mb': round(peak_mem, 2),
            'forward_memory_increase_mb': round(mem_after_forward - mem_before, 2),
            'backward_memory_increase_mb': round(mem_after_backward - mem_after_forward, 2),
        }
        
        print(f"\nMemory Usage:")
        print(f"  Before:           {results['memory_before_mb']:.2f} MB")
        print(f"  After Forward:    {results['memory_after_forward_mb']:.2f} MB (+{results['forward_memory_increase_mb']:.2f} MB)")
        print(f"  After Backward:   {results['memory_after_backward_mb']:.2f} MB (+{results['backward_memory_increase_mb']:.2f} MB)")
        print(f"  Peak Memory:      {results['peak_memory_mb']:.2f} MB")
        
        self.results['memory'] = results
        return results
    
    def profile_training_throughput(self, num_steps: int = 100, batch_size: int = 16):
        """Profile training throughput (samples/second)."""
        print(f"\n{'='*60}")
        print(f"PROFILING TRAINING THROUGHPUT ({num_steps} steps, batch_size={batch_size})")
        print(f"{'='*60}")
        
        # Create dataset and dataloader
        dataset = ToySeq2SeqDataset(
            num_samples=num_steps * batch_size,
            tokenizer=self.tokenizer,
            max_feat_len=self.config.data.max_feat_len,
            max_token_len=self.config.data.max_token_len,
            feature_dim=self.config.data.feature_dim
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer.pad_token_id)
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Warmup
        print("Warming up...")
        for i, (features, tokens, feat_lens, token_lens) in enumerate(dataloader):
            if i >= 10:
                break
            features = features.to(self.device)
            tokens = tokens.to(self.device)
            feat_lens = feat_lens.to(self.device)
            token_lens = token_lens.to(self.device)
            
            logits = self.model(features, tokens, feat_lens, token_lens)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Profile
        print(f"Profiling {num_steps} training steps...")
        self.model.train()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        total_samples = 0
        
        for i, (features, tokens, feat_lens, token_lens) in enumerate(dataloader):
            if i >= num_steps:
                break
            
            features = features.to(self.device)
            tokens = tokens.to(self.device)
            feat_lens = feat_lens.to(self.device)
            token_lens = token_lens.to(self.device)
            
            # Forward
            logits = self.model(features, tokens, feat_lens, token_lens)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_samples += features.size(0)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        throughput = total_samples / elapsed_time
        time_per_step = elapsed_time / num_steps
        
        results = {
            'num_steps': num_steps,
            'batch_size': batch_size,
            'total_samples': total_samples,
            'elapsed_time_sec': round(elapsed_time, 2),
            'throughput_samples_per_sec': round(throughput, 2),
            'time_per_step_ms': round(time_per_step * 1000, 2),
        }
        
        print(f"\nTraining Throughput:")
        print(f"  Total Samples:    {results['total_samples']}")
        print(f"  Elapsed Time:     {results['elapsed_time_sec']:.2f} sec")
        print(f"  Throughput:       {results['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"  Time per Step:    {results['time_per_step_ms']:.2f} ms")
        
        self.results['training_throughput'] = results
        return results
    
    def profile_inference_latency(self, num_runs: int = 100, batch_size: int = 1, seq_len: int = 200):
        """Profile inference latency."""
        print(f"\n{'='*60}")
        print(f"PROFILING INFERENCE LATENCY ({num_runs} runs, batch_size={batch_size})")
        print(f"{'='*60}")
        
        self.model.eval()
        
        # Create dummy input
        features = torch.randn(batch_size, seq_len, self.config.data.feature_dim).to(self.device)
        feat_lens = torch.full((batch_size,), seq_len, dtype=torch.long).to(self.device)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(10):
                _ = self.model.encode(features, feat_lens)
        
        # Profile
        print(f"Profiling {num_runs} inference runs...")
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                encoder_out, predictor_out = self.model.encode(features, feat_lens)
                decoded = self.model.decode_greedy(encoder_out, predictor_out, max_len=100)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        mean_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        results = {
            'num_runs': num_runs,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'mean_latency_ms': round(mean_latency, 2),
            'min_latency_ms': round(min_latency, 2),
            'max_latency_ms': round(max_latency, 2),
            'p50_latency_ms': round(p50_latency, 2),
            'p95_latency_ms': round(p95_latency, 2),
            'p99_latency_ms': round(p99_latency, 2),
            'throughput_samples_per_sec': round(1000 / mean_latency * batch_size, 2),
        }
        
        print(f"\nInference Latency:")
        print(f"  Mean:     {results['mean_latency_ms']:.2f} ms")
        print(f"  Min:      {results['min_latency_ms']:.2f} ms")
        print(f"  Max:      {results['max_latency_ms']:.2f} ms")
        print(f"  P50:      {results['p50_latency_ms']:.2f} ms")
        print(f"  P95:      {results['p95_latency_ms']:.2f} ms")
        print(f"  P99:      {results['p99_latency_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
        
        self.results['inference_latency'] = results
        return results
    
    def profile_model_size(self):
        """Profile model size and parameter count."""
        print(f"\n{'='*60}")
        print(f"PROFILING MODEL SIZE")
        print(f"{'='*60}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate model size in MB (assuming float32)
        model_size_mb = total_params * 4 / 1024**2
        
        # Break down by component
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        predictor_params = sum(p.numel() for p in self.model.predictor.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
        
        results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'encoder_parameters': encoder_params,
            'predictor_parameters': predictor_params,
            'decoder_parameters': decoder_params,
        }
        
        print(f"\nModel Size:")
        print(f"  Total Parameters:      {results['total_parameters']:,}")
        print(f"  Trainable Parameters:  {results['trainable_parameters']:,}")
        print(f"  Model Size:            {results['model_size_mb']:.2f} MB")
        print(f"\nComponent Breakdown:")
        print(f"  Encoder:    {results['encoder_parameters']:,} ({results['encoder_parameters']/results['total_parameters']*100:.1f}%)")
        print(f"  Predictor:  {results['predictor_parameters']:,} ({results['predictor_parameters']/results['total_parameters']*100:.1f}%)")
        print(f"  Decoder:    {results['decoder_parameters']:,} ({results['decoder_parameters']/results['total_parameters']*100:.1f}%)")
        
        self.results['model_size'] = results
        return results
    
    def save_results(self, output_path: str):
        """Save profiling results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        self.results['metadata'] = {
            'config_path': str(self.config),
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Profile model performance')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, default='results/performance_profile.json', help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for profiling')
    parser.add_argument('--num-steps', type=int, default=100, help='Number of training steps to profile')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of inference runs to profile')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"PERFORMANCE PROFILING")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    
    profiler = PerformanceProfiler(args.config, args.device)
    
    # Run all profiling tasks
    profiler.profile_model_size()
    profiler.profile_memory(batch_size=args.batch_size)
    profiler.profile_training_throughput(num_steps=args.num_steps, batch_size=args.batch_size)
    profiler.profile_inference_latency(num_runs=args.num_runs, batch_size=1)
    
    # Save results
    profiler.save_results(args.output)


if __name__ == '__main__':
    main()
