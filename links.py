import copy
from typing import Dict, List, Tuple

from bitarray import bitarray


class Links:
    __num_ocs: int
    __num_switch_per_group: int
    __num_group: int
    # (num_group, num_switch_per_group, num_ocs)   (64, 16, 512)
    __idle_links: List[List[bitarray]]
    # {job_id: [(src_group_id, src_switch_id, dst_group_id, dst_switch_id, ocs_id), ...]}
    __links_for_job: Dict[int, List[Tuple[int, int, int, int, int]]]

    def __init__(self):
        self.__num_ocs = 512
        self.__num_switch_per_group = 16
        self.__num_group = 64
        self.__idle_links = []
        self.__links_for_job = {}

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

    def get_num_idle_links(self) -> List[int]:
        return [
            sum(switch_idle_links.count(1) for switch_idle_links in group_idle_links)
            for group_idle_links in self.__idle_links
        ]

    def get_num_idle_links_fo_group(self, group_id) -> int:
        group_idle_links = self.__idle_links[group_id]
        return sum(
            [switch_idle_links.count(1) for switch_idle_links in group_idle_links]
        )

    def allocate_link_for_job(
        self, job_id, links: List[Tuple[int, int, int, int, int]]
    ):
        for src_group_id, src_switch_id, dst_group_id, dst_switch_id, ocs_id in links:
            assert self.__idle_links[src_group_id][src_switch_id][ocs_id] == 1
            assert self.__idle_links[dst_group_id][dst_switch_id][ocs_id] == 1
            self.__idle_links[src_group_id][src_switch_id][ocs_id] = 0
            self.__idle_links[dst_group_id][dst_switch_id][ocs_id] = 0
        self.__links_for_job[job_id] = links
        return False

    def free_link_for_job(self, job_id):
        links = self.__links_for_job[job_id]
        for src_group_id, src_switch_id, dst_group_id, dst_switch_id, ocs_id in links:
            assert self.__idle_links[src_group_id][src_switch_id][ocs_id] == 0
            assert self.__idle_links[dst_group_id][dst_switch_id][ocs_id] == 0
            self.__idle_links[src_group_id][src_switch_id][ocs_id] = 1
            self.__idle_links[dst_group_id][dst_switch_id][ocs_id] = 1
        del self.__links_for_job[job_id]

    def get_temp_idle_links(self) -> List[List[bitarray]]:
        return copy.deepcopy(self.__idle_links)


if __name__ == "__main__":
    links = Links()
    print(links.get_num_idle_links())
    print(links.get_num_idle_links_fo_group(0))
    links.allocate_link_for_job(1, [(0, 0, 1, 0, 0)])
    print(links.get_num_idle_links())
    links.free_link_for_job(1)
    print(links.get_num_idle_links())
