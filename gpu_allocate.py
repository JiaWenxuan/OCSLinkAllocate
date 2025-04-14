from typing import List, Tuple, Set
import numpy as np

def gpu_allocate(job_gpu_count, group):
    """
    为任务分配GPU资源并计算链路需求
    
    Args:
        job_gpu_count: 任务需要的GPU数量
        group: Groups对象，包含所有计算集群组的信息
    
    Returns:
        List[Tuple[int, int, int, int]]: 链路需求列表，每个元素是(group_id_1, group_id_2, min_links, max_links)
    """
    GROUP_COUNT = group.num_groups
    
    # 创建分配方案
    allocation = [0] * GROUP_COUNT
    remaining_gpus = job_gpu_count
    last_allocated_group = None
    allocation_success = False
    
    # 第一轮分配：遍历所有组，尝试分配GPU
    for group_id in range(GROUP_COUNT):
        if remaining_gpus <= 0:
            allocation_success = True
            break
            
        # 获取当前组信息
        current_group = group.get_group(group_id)
        available_gpus = current_group['available_gpus']
        available_links = current_group['available_links']
        
        # 跳过没有GPU的组
        if available_gpus == 0:
            continue
            
        # 如果有上一个分配的组，检查当前组是否有足够的链路
        if last_allocated_group is not None and available_links < 2:
            continue
            
        # 计算当前组可以分配多少GPU
        gpus_to_allocate = min(available_gpus, remaining_gpus)
        
        if gpus_to_allocate > 0:
            allocation[group_id] = gpus_to_allocate
            # 更新组的可用GPU数量
            group.update_available_gpus(group_id, -gpus_to_allocate)
            remaining_gpus -= gpus_to_allocate
            last_allocated_group = group_id
    
    # 如果还有未分配的GPU，继续寻找可用的组
    if remaining_gpus > 0:
        # 再次遍历，尝试在其他组中分配剩余的GPU
        for group_id in range(GROUP_COUNT):
            if remaining_gpus <= 0:
                allocation_success = True
                break
                
            # 跳过已经分配过或没有GPU的组
            if allocation[group_id] > 0:
                continue
                
            # 获取当前组信息
            current_group = group.get_group(group_id)
            available_gpus = current_group['available_gpus']
            available_links = current_group['available_links']
            
            # 跳过没有GPU的组
            if available_gpus == 0:
                continue
                
            # 检查链路连接
            if last_allocated_group is not None and available_links < 2:
                continue
            
            gpus_to_allocate = min(available_gpus, remaining_gpus)
            if gpus_to_allocate > 0:
                allocation[group_id] = gpus_to_allocate
                # 更新组的可用GPU数量
                group.update_available_gpus(group_id, -gpus_to_allocate)
                remaining_gpus -= gpus_to_allocate
                last_allocated_group = group_id
    
    # 检查是否完全分配成功
    if remaining_gpus <= 0:
        allocation_success = True
    
    # 如果分配失败，回滚已分配的资源并返回空列表
    if not allocation_success:
        # 回滚已分配的GPU
        for group_id, allocated_gpus in enumerate(allocation):
            if allocated_gpus > 0:
                group.update_available_gpus(group_id, allocated_gpus)
        return []
    
    # 计算流量矩阵和流数矩阵
    traffic_matrix = calculate_traffic_matrix(allocation)
    flow_matrix = calculate_flow_count_matrix(allocation)
    
    # 找出所有有分配GPU的组
    used_groups = {i for i, gpus in enumerate(allocation) if gpus > 0}
    
    # 如果只有一个组被分配了GPU，则无需链路
    if len(used_groups) <= 1:
        return []
    
    # 初始化链路矩阵
    best_links_matrix = np.zeros((GROUP_COUNT, GROUP_COUNT), dtype=int)
    min_links_matrix = np.zeros((GROUP_COUNT, GROUP_COUNT), dtype=int)
    
    # 计算每个集群的总带宽需求
    total_traffic = {g: 0 for g in used_groups}
    for g1 in used_groups:
        total_traffic[g1] = sum(traffic_matrix[g1])
    
    # 获取每个集群的初始可用链路数 (不受GPU使用影响的链路数)
    initial_uplinks = {g: group.get_group(g)['available_links'] for g in used_groups}
    
    # 计算组间链路分配
    for g1 in used_groups:
        for g2 in used_groups:
            if g1 < g2:  # 只处理上三角矩阵
                traffic = traffic_matrix[g1][g2]
                flow = flow_matrix[g1][g2]
                used_gpu_g1 = allocation[g1]
                used_gpu_g2 = allocation[g2]
                
                if traffic > 0:
                    # 最少链路数: 流量 > 0时至少分配1条链路
                    min_links_matrix[g1][g2] = 1
                    min_links_matrix[g2][g1] = 1
                    
                    # 获取两个组的信息
                    group1 = group.get_group(g1)
                    group2 = group.get_group(g2)
                    
                    # 计算可用链路数: 流量/100G带宽（假设100G每链路）
                    max_links = min(
                        initial_uplinks[g1] * (1 - group1['available_gpus'] / (group1['available_gpus'] + used_gpu_g1)),
                        initial_uplinks[g2] * (1 - group2['available_gpus'] / (group2['available_gpus'] + used_gpu_g2))
                    )
                    link_capacity = traffic / 100  # 单链路带宽为100G
                    
                    # 计算总带宽需求
                    total_demand = total_traffic[g1] + total_traffic[g2]
                    
                    # 按比例分配链路数
                    ratio_g1 = traffic / total_traffic[g1] if total_traffic[g1] > 0 else 0
                    ratio_g2 = traffic / total_traffic[g2] if total_traffic[g2] > 0 else 0
                    
                    # 计算最佳链路数
                    best_links = int(min(
                        max_links,
                        flow,
                        int(np.round(link_capacity)),
                        initial_uplinks[g1] * (1 - group1['available_gpus'] / (group1['available_gpus'] + used_gpu_g1)) * ratio_g1,
                        initial_uplinks[g2] * (1 - group2['available_gpus'] / (group2['available_gpus'] + used_gpu_g2)) * ratio_g2
                    ))
                    
                    # 更新最佳链路数矩阵
                    best_links_matrix[g1][g2] = best_links
                    best_links_matrix[g2][g1] = best_links
    
    # 构建链路需求列表
    link_demands = []
    for g1 in used_groups:
        for g2 in used_groups:
            if g1 < g2 and best_links_matrix[g1][g2] > 0:
                link_demands.append((g1, g2, min_links_matrix[g1][g2], best_links_matrix[g1][g2]))
    
    return link_demands

def calculate_traffic_matrix(allocation):
    """
    计算单个任务在集群间的流量矩阵，采用环形通信模式
    :param allocation: 单个任务的GPU分配方案
    :return: 该任务的流量矩阵 (单位: Gbps)
    """
    GROUP_COUNT = 64
    GPUS_PER_TRAFFIC_GROUP = 128  # 每128卡为一个流量组
    MESSAGE_SIZE = 128*100  # 每次传输128*100MB（每个数据并行组）
    BATCHES_PER_SECOND = 20  # 每秒可以计算50个batch(iteration)
    MB_TO_GBITS = 8 / 1000  # 1MB = 0.008Gb
    
    flow_matrix = [[0] * GROUP_COUNT for _ in range(GROUP_COUNT)]
    
    # 找出所有有分配GPU的组
    active_groups = [i for i, gpus in enumerate(allocation) if gpus > 0]
    if len(active_groups) <= 1:
        return flow_matrix
        
    # 计算数据并行组的数量
    total_gpus = sum(allocation)
    dp_groups = (total_gpus + GPUS_PER_TRAFFIC_GROUP - 1) // GPUS_PER_TRAFFIC_GROUP
    
    if dp_groups <= 1:
        return flow_matrix
        
    # 计算一个batch需要的通信量（单位：MB）
    base_volume = MESSAGE_SIZE * 2 * (dp_groups - 1)  # 128MB * 2 * (dp_groups-1)
    
    # 转换为每秒流量（Gbps）
    base_volume_gbps = (base_volume * BATCHES_PER_SECOND * MB_TO_GBITS)
    
    # 环形通信：记录相邻集群之间的总流量
    for i in range(len(active_groups)):
        group1 = active_groups[i]
        group2 = active_groups[(i + 1) % len(active_groups)]
        
        # 更新流量矩阵（双向相同）
        flow_matrix[group1][group2] += base_volume_gbps
        flow_matrix[group2][group1] += base_volume_gbps
    
    return flow_matrix

def calculate_flow_count_matrix(allocation):
    """
    计算单个任务在集群间的流数矩阵，采用环形通信模式
    :param allocation: 单个任务的GPU分配方案
    :return: 该任务的流数矩阵 (单位: flows/s)
    """
    GROUP_COUNT = 64
    GPUS_PER_COMM_GROUP = 128  # 每128卡为一个通信组
    BATCHES_PER_SECOND = 50  # 每秒可以计算50个batch
    
    flow_matrix = [[0] * GROUP_COUNT for _ in range(GROUP_COUNT)]
    
    # 找出所有有分配GPU的组
    active_groups = [i for i, gpus in enumerate(allocation) if gpus > 0]
    if len(active_groups) <= 1:
        return flow_matrix
        
    # 计算数据并行组的数量
    total_gpus = sum(allocation)
    dp_groups = (total_gpus + GPUS_PER_COMM_GROUP - 1) // GPUS_PER_COMM_GROUP
    
    if dp_groups <= 1:
        return flow_matrix
    
    # 每个batch产生2*(dp_groups-1)个流
    flows_per_batch = 2 * (dp_groups - 1)
    flows_per_second = flows_per_batch * BATCHES_PER_SECOND
    
    # 环形通信：记录相邻集群之间的流数
    for i in range(len(active_groups)):
        group1 = active_groups[i]
        group2 = active_groups[(i + 1) % len(active_groups)]
        
        # 更新流数矩阵（双向相同）
        flow_matrix[group1][group2] += flows_per_second
        flow_matrix[group2][group1] += flows_per_second
    
    return flow_matrix
