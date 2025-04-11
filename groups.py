class Groups:
    def __init__(self):
        self.num_groups = 64
        self.groups = []
        for group_id in range(self.num_groups):
            self.groups.append({
                'group_id': group_id,
                'available_gpus': 2048,  # 初始可用GPU数量
                'available_links': 2048   # 初始可用链路数量
            })

    def get_group(self, group_id):
        """获取指定group的信息"""
        return self.groups[group_id]

    def update_available_gpus(self, group_id, delta):
        """更新指定group的可用GPU数量"""
        self.groups[group_id]['available_gpus'] += delta

    def update_available_links(self, group_id, delta):
        """更新指定group的可用链路数量"""
        self.groups[group_id]['available_links'] += delta