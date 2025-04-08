class Job:
    def __init__(self, gpu_count, time):
        self.gpu_count = gpu_count
        self.time = time


class Jobs:
    def __init__(self):
        # 可选的GPU数量及其对应的时间
        gpu_time_options = [
            (512, 1), (512, 1), (512, 1),    # 512 GPU jobs take 1 time unit
            (1024, 2), (1024, 2), (1024, 2),  # 1024 GPU jobs take 2 time units
            (2048, 3),                         # 2048 GPU jobs take 3 time units
            (4096, 4),                         # 4096 GPU jobs take 4 time units
            (8192, 5),                         # 8192 GPU jobs take 5 time units
            (16384, 6)                         # 16384 GPU jobs take 6 time units
        ]
        # 生成200个随机选择的任务
        self.jobs = [Job(gpu_count, time) for gpu_count, time in [random.choice(gpu_time_options) for _ in range(200)]]
        self.current_index = 0

    def get_next_job(self):
        """获取下一个任务，如果还有任务则返回任务对象，否则返回None"""
        if self.current_index < len(self.jobs):
            job = self.jobs[self.current_index]
            return job
        return None

    def has_more_jobs(self):
        """检查是否还有更多任务"""
        return self.current_index < len(self.jobs)

    def get_remaining_jobs_count(self):
        """获取剩余任务数量"""
        return len(self.jobs) - self.current_index
