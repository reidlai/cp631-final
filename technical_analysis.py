import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void ema(float *values, float *ema_values, int days, int n, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float alpha = 2.0f / (days + 1);
        for (int j = 0; j < m; j++) {
            if (j == 0) {
                ema_values[idx * m] = values[idx * m];  // start with the first value
            } else {
                ema_values[idx * m + j] = alpha * values[idx * m + j] + (1 - alpha) * ema_values[idx * m + j - 1];
            }
        }
    }
}

__global__ void compute_gains_losses(float *values, float *gains, float *losses, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float change = values[idx] - values[idx - 1];
        gains[idx] = change > 0 ? change : 0;
        losses[idx] = change < 0 ? -change : 0;
    }
}

__global__ void macd(float *values, float *macd_values, float *signal_values, int short_period, int long_period, int signal_period, int n, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float alpha_short = 2.0f / (short_period + 1);
        float alpha_long = 2.0f / (long_period + 1);
        float alpha_signal = 2.0f / (signal_period + 1);
        float ema_short = 0;
        float ema_long = 0;
        float ema_signal = 0;
        for (int j = 0; j < m; j++) {
            if (j < short_period) {
                ema_short = alpha_short * values[idx * m + j] + (1 - alpha_short) * ema_short;
            }
            if (j < long_period) {
                ema_long = alpha_long * values[idx * m + j] + (1 - alpha_long) * ema_long;
            }
            float macd = ema_short - ema_long;
            if (j < signal_period) {
                ema_signal = alpha_signal * macd + (1 - alpha_signal) * ema_signal;
            }
            macd_values[idx * m + j] = macd;
            signal_values[idx * m + j] = ema_signal;
        }
    }
}
""")

def ema_gpu(values, days):
    n, m = values.shape
    ema_values = np.empty_like(values)
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    func = mod.get_function("ema")
    func(cuda.In(values), cuda.Out(ema_values), np.int32(days), np.int32(n), np.int32(m), block=(block_size,1,1), grid=(grid_size,1))
    return ema_values

def ema(days, values):
    alpha = 2 / (days + 1)
    ema_values = [values[0]]  # start with the first value
    for value in values[1:]:
        ema_values.append(alpha * value + (1 - alpha) * ema_values[-1])
    return ema_values[-1]

def rsi_gpu(days, values):
    n = len(values)
    gains = np.empty_like(values)
    losses = np.empty_like(values)
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    func = mod.get_function("compute_gains_losses")
    func(cuda.In(values), cuda.Out(gains), cuda.Out(losses), np.int32(n), block=(block_size,1,1), grid=(grid_size,1))

    avg_gain = np.sum(gains[:days]) / days
    avg_loss = np.sum(losses[:days]) / days
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi_value = 100 - (100 / (1 + rs))
    return rsi_value

def rsi(days, values):
    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)
    avg_gain = sum(gains[:days]) / days
    avg_loss = sum(losses[:days]) / days
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi_value = 100 - (100 / (1 + rs))
    return rsi_value

def macd_gpu(values, short_period=12, long_period=26, signal_period=9):
    n, m = values.shape
    macd_values = np.empty_like(values)
    signal_values = np.empty_like(values)
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    func = mod.get_function("macd")
    func(cuda.In(values), cuda.Out(macd_values), cuda.Out(signal_values), np.int32(short_period), np.int32(long_period), np.int32(signal_period), np.int32(n), np.int32(m), block=(block_size,1,1), grid=(grid_size,1))
    return macd_values, signal_values

def macd(values, short_period=12, long_period=26, signal_period=9):
    ema_short = exponential_moving_average(short_period, values)
    ema_long = exponential_moving_average(long_period, values)
    macd_line = np.array(ema_short) - np.array(ema_long)
    signal_line = exponential_moving_average(signal_period, macd_line.tolist())
    return macd_line, signal_line