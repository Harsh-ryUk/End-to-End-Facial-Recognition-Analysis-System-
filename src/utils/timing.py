import time
import cv2

class FPSMeter:
    def __init__(self):
        self.prev_time = time.time()
        self.curr_time = 0
        self.fps = 0
        self.frame_count = 0
        self.accum_time = 0

    def update(self):
        self.curr_time = time.time()
        dt = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        
        self.accum_time += dt
        self.frame_count += 1
        
        if self.accum_time >= 1.0:
            self.fps = self.frame_count / self.accum_time
            self.frame_count = 0
            self.accum_time = 0
            
        return self.fps, dt * 1000 # FPS, Latency(ms)

class LatencyTracker:
    def __init__(self):
        self.start_times = {}
        self.latencies = {}
        
    def start(self, name):
        self.start_times[name] = time.perf_counter()
        
    def end(self, name):
        if name in self.start_times:
            dt = (time.perf_counter() - self.start_times[name]) * 1000
            self.latencies[name] = dt
            return dt
        return 0
        
    def get_latency(self):
        return self.latencies
