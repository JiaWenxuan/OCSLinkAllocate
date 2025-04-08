class Groups:
    pass


class Links:
    def __init__(self, id, capacity):
        self.__num_ocs = 512
        self.__num_switch_per_group = 16
        self.__num_group = 64
        self.__idle_links = []  # (num_group, num_switch_per_group, num_ocs)   (64, 16, 512)
        spine_0 = bitarray(512)
        spine_0[:128] = 1  # low index in left side
        spine_1 = bitarray(512)
        spine_1[128:256] = 1  # low index in left side
        spine_2 = bitarray(512)
        spine_2[256:384] = 1  # low index in left side
        spine_3 = bitarray(512)
        spine_3[384:512] = 1  # low index in left side
        for _ in range(self.__num_group):
            group_idle_links = []
            for switch_id in range(self.__num_switch_per_group):
                if switch_id % 4 == 0:
                    group_idle_links.append(spine_0.copy())
                elif switch_id % 4 == 1:
                    group_idle_links.append(spine_1.copy())
                elif switch_id % 4 == 2:
                    group_idle_links.append(spine_2.copy())
                else:
                    group_idle_links.append(spine_3.copy())
            self.__idle_links.append(group_idle_links)


def allocate_test():
    jobs = get_jobs()
    job_index = 0
    time = 0
    while job_index < len(jobs):
        free_job_arrive_time()
        while True:
            job = jobs[job_index]
            a = gpu_allocate(job, group)
            if not a:
                break
            b = classa.logic_link_allocate(job, Links, a)
            if not b:
                break
            c = physical_link_allocate(job, Links, b)
            if not c:
                break
            print(f"Job {job_index} allocated at time {time}")
            job_index += 1
    time += 1


if __name__ == "__main__":
    allocate_test()
