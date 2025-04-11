from typing import List, Tuple

from links import Links
from physical_link_allocate import (
    AllocationResult,
    PhysicalLinkeAllocatingAlthrighm,
    physical_link_allocate,
)


class Groups:
    pass


def allocate_test():
    links = Links()

    jobs = get_jobs()
    job_index = 0
    time = 0
    while job_index < len(jobs):
        free_job_arrive_time()
        while True:
            job = jobs[job_index]
            # link_demand : [(group_id_1, group_id_2, min_links, max_links), ...]
            link_demand: List[Tuple[int, int, int, int]] = gpu_and_logic_link_allocate(
                job, links
            )
            if not link_demand:
                break
            used_kinks = physical_link_allocate(temp_idle_links, link_demand)
            if not used_kinks:
                break
            links.allocate_link_for_job(job_index, used_kinks)
            print(f"Job {job_index} allocated at time {time}")
            job_index += 1
    time += 1


if __name__ == "__main__":
    allocate_test()
