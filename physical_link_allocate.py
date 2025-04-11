from enum import IntEnum
from typing import List, Tuple

from bitarray import bitarray


class PhysicalLinkeAllocatingAlthrighm(IntEnum):
    NATIVE = 0
    OPTIMIZE = 1


class AllocationResult(IntEnum):
    MEETMAX = 0
    MEETMIN = 1
    FAILURE = 2


def physical_link_allocate(
    temp_idle_links: List[List[bitarray]],
    link_demand: List[Tuple[int, int, int, int]],
    used_links: List[Tuple[int, int, int, int, int]],
    alghrithm: PhysicalLinkeAllocatingAlthrighm,
) -> AllocationResult:
    if alghrithm == PhysicalLinkeAllocatingAlthrighm.NATIVE:
        return physical_link_allocate_native(temp_idle_links, link_demand, used_links)
    elif alghrithm == PhysicalLinkeAllocatingAlthrighm.OPTIMIZE:
        return physical_link_allocate_optimize(temp_idle_links, link_demand, used_links)
    else:
        raise ValueError("Invalid algorithm selected")


def physical_link_allocate_optimize(
    temp_idle_links: List[List[bitarray]],
    link_demand: List[Tuple[int, int, int, int]],
    used_links: List[Tuple[int, int, int, int, int]],
) -> AllocationResult:
    result = AllocationResult.MEETMAX
    used_links = []
    for i in range(2):
        for group_id_1, group_id_2, min_demand, max_demand in link_demand:
            demand = min_demand if i == 0 else max_demand
            if demand == 0:
                continue
            if not allocate_for_two_groups_optimize(
                (group_id_1, group_id_2), demand, temp_idle_links, used_links
            ):
                if i == 0:
                    return AllocationResult.FAILURE
                else:
                    result = AllocationResult.MEETMIN
    return result


def allocate_for_two_groups_optimize(
    group_id_pair, num_link, temp_idle_links, used_links
):
    num_switch_per_group = len(temp_idle_links[group_id_pair[0]])
    src_index = 0
    dst_index = 0
    src_to_dst = [
        [True for _ in range(num_switch_per_group)] for _ in range(num_switch_per_group)
    ]
    links = 0
    while links < num_link:
        # find a enable link
        while True:
            for _ in range(num_switch_per_group):
                for _ in range(num_switch_per_group):
                    if src_to_dst[src_index][dst_index]:
                        break
                    else:
                        dst_index = (dst_index + 1) % num_switch_per_group
                if src_to_dst[src_index][dst_index]:
                    break
                else:
                    src_index = (src_index + 1) % num_switch_per_group
            else:
                # if not self.CheckNoLink(temp_idle_links, group_id_pair):
                #     raise Exception("still have link")
                return False

            enable_links = (
                temp_idle_links[group_id_pair[0]][src_index]
                & temp_idle_links[group_id_pair[1]][dst_index]
            )
            try:
                first_enable_link = enable_links.index(1)
                break
            except ValueError:
                src_to_dst[src_index][dst_index] = False
                continue

        # allocate link
        temp_idle_links[group_id_pair[0]][src_index][first_enable_link] = 0
        temp_idle_links[group_id_pair[1]][dst_index][first_enable_link] = 0
        used_links.append(
            (
                group_id_pair[0],
                src_index,
                group_id_pair[1],
                dst_index,
                first_enable_link,
            )
        )
        src_index = (src_index + 1) % num_switch_per_group
        dst_index = (dst_index + 1) % num_switch_per_group
        links += 1
    return True


def allocate_for_two_groups_native(
    group_id_pair, num_link, temp_idle_links, used_links
):
    links = 0
    num_ocs = len(temp_idle_links[group_id_pair[0]][0])
    num_switch_per_group = len(temp_idle_links[group_id_pair[0]])
    for switch_id_1 in range(num_switch_per_group):
        for switch_id_2 in range(num_switch_per_group):
            for ocs_id in range(num_ocs):
                if (
                    temp_idle_links[group_id_pair[0]][switch_id_1][ocs_id] == 1
                    and temp_idle_links[group_id_pair[1]][switch_id_2][ocs_id] == 1
                ):
                    temp_idle_links[group_id_pair[0]][switch_id_1][ocs_id] = 0
                    temp_idle_links[group_id_pair[1]][switch_id_2][ocs_id] = 0
                    links = links + 1
                    used_links.append(
                        (
                            group_id_pair[0],
                            switch_id_1,
                            group_id_pair[1],
                            switch_id_2,
                            ocs_id,
                        )
                    )
                    if links == num_link:
                        return True
    return False


def physical_link_allocate_native(
    temp_idle_links: List[List[bitarray]],
    link_demand: List[Tuple[int, int, int, int]],
    used_links: List[Tuple[int, int, int, int, int]],
) -> AllocationResult:
    result = AllocationResult.MEETMAX
    used_links = []
    for group_id_1, group_id_2, _, demand in link_demand:
        if demand == 0:
            continue
        if not allocate_for_two_groups_native(
            (group_id_1, group_id_2),
            demand,
            temp_idle_links,
            used_links,
        ):
            return AllocationResult.FAILURE
    return result
