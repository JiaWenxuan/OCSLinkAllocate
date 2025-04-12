from typing import List, Tuple

from groups import Groups
from jobs import Jobs
from links import Links
from physical_link_allocate import (
    AllocationResult,
    PhysicalLinkeAllocatingAlthrighm,
    physical_link_allocate,
)


def allocate_test():
    links = Links()
    group = Groups()
    jobs = Jobs()
    time = 0
    while jobs.has_more_jobs():
        jobs_by_time = jobs.get_jobs_by_time(time)
        for job in jobs_by_time:
            links.free_link_for_job(job)
            group.free_gpu_for_job(job)
            job.finish()

        while True:
            job = jobs.get_next_job()
            if job is None:
                break
            # link_demand : [(group_id_1, group_id_2, min_links, max_links), ...]
            link_demand: List[Tuple[int, int, int, int]] = gpu_allocate(
                job.gpu_count, group
            )
            if not link_demand:
                break
            used_kinks = []
            allocation_result = physical_link_allocate(
                links.get_temp_idle_links(),
                link_demand,
                used_kinks,
                PhysicalLinkeAllocatingAlthrighm.OPTIMIZE,
            )
            if allocation_result == AllocationResult.FAILURE:
                break
            else:
                links.allocate_link_for_job(job, used_kinks)

            print(
                f"Job {jobs.current_index} allocated at time {time}, remaining jobs: {jobs.get_remaining_jobs_count()}"
            )
            jobs.current_index += 1
        time += 1


if __name__ == "__main__":
    allocate_test()
