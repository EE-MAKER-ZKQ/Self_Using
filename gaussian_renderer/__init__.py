# import torch
# import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh
# # 必须引入 knn
# from torch_cluster import knn

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
#             override_color = None, is_train=False, iteration=None):
#     """
#     Render the scene. 
#     """
 
#     # Create zero tensor
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug,
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # --- 准备 scales, rotations, shs 等 (保持原样) ---
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # --- 修改后的 DropGaussian (KNN 3x3 区域版本) ---
    
#     if is_train and iteration is not None:
#         if ( iteration%50==0):
#             num_points = opacity.shape[0]
#             # 设定一个区域的大小，k=9 模拟 3x3 局部范围
#             k_size = 10
#             drop_rate = 0.2 * (iteration / 10000)

#             # 初始化补偿因子
#             compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")

#             if drop_rate > 0:
#                 # 计算需要撒多少个“种子”来产生对应的丢弃比例
#                 num_seeds = int((num_points * drop_rate) / k_size)

#                 if num_seeds > 0:
#                     # 随机选中心点
#                     seed_idx = torch.randint(0, num_points, (num_seeds,), device="cuda")

#                     # 找出这些中心点及其周围的邻居 (使用已经获取的 means3D)
#                     # knn 返回 [2, num_seeds * k_size]，取 [1] 获得邻居索引
#                     neighbor_idx = knn(means3D, means3D[seed_idx], k=k_size)[1]

#                     # 将这些点的补偿系数置零
#                     compensation[neighbor_idx] = 0.0

#                     # 手动执行能量补偿 (Scale)，防止渲染结果随 drop_rate 增加而变暗
#                     keep_rate = 1.0 - drop_rate
#                     compensation = compensation / (keep_rate + 1e-7)
#              # Apply to opacity
#             opacity = opacity * compensation[:, None]
#         else: 
#             compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

#             # Apply DropGaussian with compensation
#             drop_rate = 0.2 * (iteration/10000)
#             d = torch.nn.Dropout(p=drop_rate)
#             compensation = d(compensation)
#             opacity = opacity * compensation[:, None]

       

#     # --- 渲染输出 ---
#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     rendered_image = rendered_image.clamp(0, 1)
#     out = {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter" : (radii > 0).nonzero(),
#         "radii": radii
#     }
    
#     return out

# import torch
# import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh
# # 引入 knn 模块
# from torch_cluster import knn

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
#            override_color = None, is_train=False, iteration=None):
#     """
#     Render the scene. 
#     """
 
#     # Create zero tensor.
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug,
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # --- 保持原有的 scales, rotations, shs 逻辑 ---
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color
# #205-294 SOTA
#     # =========================================================
#     #  基于 KNN 密度的结构化 Dropout 逻辑
#     # =========================================================
#     if is_train and iteration is not None:
#         # 1. 初始化 compensation
#         num_points = opacity.shape[0]
        
#         # 使用缓存机制：如果是在 render 类外部，建议把 cache 存在 self 里。
#         # 这里为了演示，我们假设每 500 次计算并应用一次，中间迭代如果不存缓存会导致闪烁。
#         # 既然是 Dropout，通常每一帧都需要掩码。
#         # 逻辑：每 500 次更新一次“黑名单”，在接下来的 500 次里一直沿用这个黑名单？
#         # 或者：仅在 iteration % 500 == 0 的那一瞬间进行剪枝？
#         # 根据常规理解，这里实现为：每 500 轮更新一次遮罩并缓存（模拟持续的抑制），或者仅在特定迭代执行。
#         # 下面实现的是：每 500 轮重新计算一次遮罩，并将其存入 GaussianModel (pc) 中作为临时缓存，以供后续使用。
        
#         # 检查 pc 是否有缓存，没有则初始化全 1
#         if not hasattr(pc, "density_mask_cache") or pc.density_mask_cache.shape[0] != num_points:
#              pc.density_mask_cache = torch.ones(num_points, dtype=torch.float32, device="cuda")

#         # 每 100 次迭代更新一次掩码逻辑
#         if iteration % 100 == 0:
#             k_density = 11  # k=11 是因为包含点自己 (1+10)
#             k_drop = 10     # 要消除周围的 10 个点

#             # --- A. 计算密度 (Mean Distance to 10 neighbors) ---
#             # 为了节省显存，使用 detach 的坐标
#             curr_xyz = means3D.detach()
            
#             # knn 返回 [2, N*k]，row0 是 query(src), row1 是 neighbor(target)
#             assign_idx = knn(curr_xyz, curr_xyz, k=k_density)
            
#             # 计算欧氏距离
#             diff = curr_xyz[assign_idx[0]] - curr_xyz[assign_idx[1]]
#             sq_dist = (diff ** 2).sum(dim=1) # [N*k]
            
#             # 变形为 [N, k] 并求平均距离 (越小代表越密)
#             # 注意：knn 的输出通常是按 query 索引分组的，但不保证顺序完全严格，
#             # 为保险起见可以使用 scatter_mean，但在 dense 场景下 view 也是常用的近似。
#             # 这里假设 knn 输出是规则的。
#             sq_dist_per_point = sq_dist.view(num_points, k_density)
#             # 排除自己 (距离为0的点)，取剩下10个的均值
#             mean_dist = sq_dist_per_point[:, 1:].mean(dim=1) 

#             # --- B. 排序并筛选 Top 2% 最密集的点 ---
#             # argsort: 从小到大排序 (距离小=密度大)
#             sorted_indices = torch.argsort(mean_dist) 
            
#             # 选取 Top 2%
#             num_seeds = int(num_points * 0.01 * (iteration/10000))
#             # 候选种子点 (密度最大的点)
#             candidate_seeds = sorted_indices[:num_seeds]

#             # --- C. 区域掩码与去重 (Avoid redundant elimination) ---
#             # 你的需求：如果种子点 A 和 种子点 B 互为邻居，避免重复消除。
#             # 解决方法：
#             # 1. 找出所有候选种子的邻居。
#             # 2. 使用 unique() 将重复涉及的点合并，这样既实现了“区域置零”，又解决了“互相属于”导致的重复计算问题。
#             #    (在向量化计算中，直接对邻居取并集等效于你要求的“避免重复消除”并最大化覆盖)。
            
#             # 找出候选种子周围的 10 个邻居
#             seed_neighbors = knn(curr_xyz, curr_xyz[candidate_seeds], k=k_drop + 1)
#             # seed_neighbors[1] 包含了种子自己及其邻居的索引
#             indices_to_drop = seed_neighbors[1]
            
#             # 去重：得到最终需要置零的所有点的唯一索引
#             unique_drop_indices = torch.unique(indices_to_drop)

#             # 更新缓存：先全置 1
#             pc.density_mask_cache.fill_(1.0)
#             # 将被选中的区域置 0
#             pc.density_mask_cache[unique_drop_indices] = 0.0
            
#             # (可选) 打印调试信息，确认删了多少点
#             # print(f"Iter {iteration}: Dropping {len(unique_drop_indices)} points based on density.")

#             # 获取当前的 compensation
#             compensation = pc.density_mask_cache

#             # 应用到 Opacity
#             opacity = opacity * compensation[:, None]
#         else:
#             compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

#             # Apply DropGaussian with compensation
#             drop_rate = 0.4 * (iteration/10000)
#             d = torch.nn.Dropout(p=drop_rate)
#             compensation = d(compensation)

#             # Apply to opacity
#             opacity = opacity * compensation[:, None]
#####12.24     if is_train and iteration is not None:
#             num_points = opacity.shape[0]

#             # 1. 初始化缓存容器
#             if not hasattr(pc, "knn_map_cache"):
#                 pc.knn_map_cache = None

#             # --- [Phase A] 每 100 次迭代：重算 KNN，建立缓存，执行密度消除 ---
#             if iteration % 100 == 0:
#                 k_density = 11  # 1个点 + 10个邻居

#                 # 为了节省显存，使用 detach 的坐标
#                 curr_xyz = means3D.detach()

#                 # 1. 运行耗时的 KNN 计算
#                 assign_idx = knn(curr_xyz, curr_xyz, k=k_density)

#                 # 2. 【核心】保存 KNN 结果到缓存，供 else 复用
#                 # 将 [2, N*k] 转为 [N, k] 的查表矩阵
#                 # pc.knn_map_cache[i] 存储了点 i 的 10 个邻居索引
#                 pc.knn_map_cache = assign_idx[1].view(num_points, k_density)

#                 # 3. 计算密度 (用于本次的密度消除)
#                 # 利用刚算出的索引获取邻居坐标
#                 neighbor_xyz = curr_xyz[pc.knn_map_cache] 
#                 # 计算距离: [N, k]
#                 sq_dist = ((curr_xyz.unsqueeze(1) - neighbor_xyz) ** 2).sum(dim=-1)
#                 # 平均距离 (越小越密)
#                 mean_dist = sq_dist[:, 1:].mean(dim=1) 

#                 # 4. 筛选 Top 2% 最密集的点 (Density Selection)
#                 num_seeds = int(num_points * 0.05 * min(1.0, iteration/10000.0))

#                 if num_seeds > 0:
#                     # 选密度最大的种子
#                     sorted_indices = torch.argsort(mean_dist) 
#                     candidate_seeds = sorted_indices[:num_seeds]

#                     # 查表获取要消除的邻居
#                     neighbors_to_drop = pc.knn_map_cache[candidate_seeds].flatten()

#                     # 制作遮罩
#                     unique_drop = torch.unique(neighbors_to_drop)
#                     compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                     compensation[unique_drop] = 0.0
#                 else:
#                     compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")

#             # --- [Phase B] 其他迭代：复用 KNN 缓存 + 随机种子消除 ---
#             else:
#                 # 1. 检查缓存是否可用 (防止点数变化导致索引越界)
#                 if (pc.knn_map_cache is not None) and (pc.knn_map_cache.shape[0] == num_points):

#                     # 2. 计算需要多少个随机种子
#                     # drop_rate 随训练增加
#                     current_drop_rate = 0.05 * (iteration / 10000.0)
#                     # 除以 11 是因为每个种子会带走自己和10个邻居
#                     num_seeds = int((num_points * current_drop_rate) / 11)

#                     if num_seeds > 0:
#                         # 3. 【随机】选取种子点
#                         random_seeds = torch.randint(0, num_points, (num_seeds,), device="cuda")

#                         # 4. 【复用】查表获取邻居 (无需运行 KNN)
#                         neighbors_to_drop = pc.knn_map_cache[random_seeds].flatten()

#                         # 5. 制作临时遮罩
#                         unique_drop = torch.unique(neighbors_to_drop)
#                         compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                         compensation[unique_drop] = 0.0

#                         # 6. 【能量补偿】(Rescaling)
#                         # 随机丢弃需要保持亮度一致，密度消除(Phase A)通常不需要
#                         keep_ratio = 1.0 - (len(unique_drop) / num_points)
#                         compensation = compensation / (keep_ratio + 1e-7)
#                     else:
#                         compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                 else:
#                     # 缓存失效时的兜底
#                     compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")

#             # 应用最终计算出的 compensation (来自 if 或 else)
#             opacity = opacity * compensation[:, None]        

#     # =========================================================

#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     rendered_image = rendered_image.clamp(0, 1)
#     out = {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter" : (radii > 0).nonzero(),
#         "radii": radii
#     }
    
#     return out    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # DropGaussian
#     if is_train:
#         # --- 配置参数 ---
#         # 每次渲染时随机挖洞的数量 (建议 1-3 个)
#         num_holes = torch.randint(1, 4, (1,)).item() 
        
#         # 动态调整每个洞的大小：随着训练进行，挖的洞从 0.5% 增长到 5%
#         # 如果 iteration 未提供，默认使用较小的值
#         current_iter = iteration if iteration is not None else 0
#         progress = min(current_iter / 10000.0, 1.0) # 假设 30k 是最大步数
#         base_ratio = 0.005 + (0.045 * progress)     # 范围 [0.5%, 5%]
        
#         total_points = means3D.shape[0]
#         # 初始化掩码为 1 (全保留)
#         mask = torch.ones(total_points, device="cuda", dtype=torch.float32)
        
#         # 只有当点数足够时才执行
#         if total_points > 1000:
#             # 循环生成多个洞
#             for _ in range(num_holes):
#                 # 1. 确定当前洞要剔除的点数 (K)
#                 k_neighbors = int(total_points * base_ratio)
#                 if k_neighbors <= 0: continue

#                 # 2. 随机选择中心点 (Anchor)
#                 center_idx = torch.randint(0, total_points, (1,), device="cuda")
#                 center_point = means3D[center_idx] # (1, 3)

#                 # 3. 计算距离 (使用平方欧氏距离，避免开根号，速度更快)
#                 # means3D: (N, 3), center_point: (1, 3) -> Broadcasting
#                 dists_sq = torch.sum((means3D - center_point)**2, dim=1)

#                 # 4. 找到最近的 K 个点的索引
#                 # largest=False 表示找最小距离
#                 _, knn_indices = torch.topk(dists_sq, k=k_neighbors, largest=False)

#                 # 5. 更新掩码：将这些区域的点置为 0
#                 mask[knn_indices] = 0.0
            
#             # 6. 应用掩码到 Opacity
#             # opacity: (N, 1), mask: (N,) -> 需要 unsqueeze
#             opacity = opacity * mask.unsqueeze(1)



# import torch
# import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh
# from torch_cluster import knn

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
#            override_color = None, is_train=False, iteration=None):
#     """
#     Render the scene. 
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug,
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None

#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     # SH -> RGB conversion
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # =================================================================================
#     #  DropGaussian 综合逻辑：KNN结构化硬掩码 + 深度感知随机丢弃
#     # =================================================================================
#     if is_train and iteration is not None:
#         num_points = opacity.shape[0]
        
#         # --- 1. KNN结构化硬掩码（每 100 次迭代更新一次） ---
#         # 初始化或检查缓存
#         if not hasattr(pc, "density_mask_cache") or pc.density_mask_cache.shape[0] != num_points:
#             pc.density_mask_cache = torch.ones(num_points, dtype=torch.float32, device="cuda")

#         if iteration % 100 == 0:
#             k_density = 11  # 1中心点 + 10邻居
#             k_drop = 10     # 消除周围10个点
#             curr_xyz = means3D.detach()
            
#             # 计算局部密度（平均距离）
#             assign_idx = knn(curr_xyz, curr_xyz, k=k_density)
#             diff = curr_xyz[assign_idx[0]] - curr_xyz[assign_idx[1]]
#             sq_dist = (diff ** 2).sum(dim=1).view(num_points, k_density)
#             mean_dist = sq_dist[:, 1:].mean(dim=1) 
            
#             # 这里的 mean_dist 越小，密度越大，存入 pc 供后续随机丢弃使用
#             pc.density_score = mean_dist 

#             # 筛选 Top 2% 最密集的点作为“种子”进行硬消除
#             num_seeds = int(num_points * 0.02 * min(1.0, iteration/10000.0))
#             if num_seeds > 0:
#                 sorted_indices = torch.argsort(mean_dist) # 距离从小到大
#                 candidate_seeds = sorted_indices[:num_seeds]
                
#                 # 寻找这些种子周围的邻居
#                 indices_to_drop = knn(curr_xyz, curr_xyz[candidate_seeds], k=k_drop + 1)[1]
#                 # 去重合并区域，避免重复计算
#                 unique_drop_indices = torch.unique(indices_to_drop)

#                 pc.density_mask_cache.fill_(1.0)
#                 pc.density_mask_cache[unique_drop_indices] = 0.0
        
#         # 先应用硬掩码（直接关掉特定高密度区域）
#         opacity = opacity * pc.density_mask_cache[:, None]

#         # --- 2. 深度感知随机丢弃（每一帧执行，基于概率） ---
#         # A. 计算相机空间深度 (Z-depth)
#         ones = torch.ones((num_points, 1), device="cuda")
#         gaussian_positions_homo = torch.cat([means3D, ones], dim=1)
#         camera_coordinates = torch.matmul(gaussian_positions_homo, viewpoint_camera.world_view_transform.T)
#         camera_depths = camera_coordinates[:, 2] # 取 Z 轴
        
#         depth_min, depth_max = camera_depths.min(), camera_depths.max()
#         # 归一化深度得分：近处得分高
#         depth_score = (1.0 - (camera_depths - depth_min) / (depth_max - depth_min + 1e-6)).float()
        
#         # B. 深度分带 (Near/Mid/Far)
#         sorted_depths, _ = torch.sort(camera_depths)
#         depth_percentile_33 = sorted_depths[int(num_points * 0.33)].float()
#         depth_percentile_67 = sorted_depths[int(num_points * 0.67)].float()
        
#         near_field = camera_depths <= depth_percentile_33
#         mid_field = (camera_depths > depth_percentile_33) & (camera_depths <= depth_percentile_67)
#         far_field = camera_depths > depth_percentile_67

#         # C. 结合密度分数 (使用第1步中 KNN 算出的密度)
#         density_norm = torch.ones_like(depth_score) * 0.5
#         if hasattr(pc, 'density_score') and pc.density_score.numel() >= num_points:
#             # 密度越大（mean_dist越小），分数应该越高。所以用 1 - 归一化值
#             d_dist = pc.density_score[:num_points].float()
#             d_min, d_max = d_dist.min(), d_dist.max()
#             density_norm = 1.0 - ((d_dist - d_min) / (d_max - d_min + 1e-6))

#         # D. 计算最终丢弃概率
#         depth_weight, density_weight = 0.5, 0.5  # 权重可调
#         drop_min, drop_max = 0.05, 0.2          # 基础丢弃率范围
        
#         combined_score = (depth_weight * depth_score + density_weight * density_norm).float()
#         progress = min(1.0, iteration / 10000.0)
#         current_base_rate = float(drop_min + (drop_max - drop_min) * progress)
        
#         # 不同场次的衰减
#         drop_prob = (near_field.float() * combined_score * current_base_rate + 
#                      mid_field.float() * combined_score * current_base_rate * 0.7 + 
#                      far_field.float() * combined_score * current_base_rate * 0.3)
        
#         keep_prob = 1.0 - drop_prob
#         # 随机采样生成掩码
#         random_mask = (torch.rand_like(keep_prob) < keep_prob).float()
        
#         # 最终应用概率掩码
#         opacity = opacity * random_mask[:, None]
#     if is_train and iteration is not None:
#         num_points = opacity.shape[0]

#         # =========================================================
#         # 1. KNN 结构化硬掩码 (每 100 次迭代更新)
#         # =========================================================
#         # 初始化缓存
#         if not hasattr(pc, "density_mask_cache") or pc.density_mask_cache.shape[0] != num_points:
#             pc.density_mask_cache = torch.ones(num_points, dtype=torch.float32, device="cuda")

#         if iteration % 100 == 0:
#             k_density = 11
#             curr_xyz = means3D.detach()

#             # 计算密度：利用最近邻平均距离
#             assign_idx = knn(curr_xyz, curr_xyz, k=k_density)
#             diff = curr_xyz[assign_idx[0]] - curr_xyz[assign_idx[1]]
#             sq_dist = (diff ** 2).sum(dim=1).view(num_points, k_density)
#             # 存入 pc 供第二步概率计算复用
#             pc.density_score = sq_dist[:, 1:].mean(dim=1) 

#             # 选 Top 2% 最密集的种子点
#             num_seeds = int(num_points * 0.02 * min(1.0, iteration / 10000.0))
#             if num_seeds > 0:
#                 sorted_indices = torch.argsort(pc.density_score) # 距离小 = 密度大
#                 candidate_seeds = sorted_indices[:num_seeds]

#                 # 找到种子周围的邻居并去重
#                 neighbor_idx = knn(curr_xyz, curr_xyz[candidate_seeds], k=11)[1]
#                 pc.density_mask_cache.fill_(1.0)
#                 pc.density_mask_cache[torch.unique(neighbor_idx)] = 0.0

#         # 应用 KNN 结构化掩码 (Hard Mask)
#         opacity = opacity * pc.density_mask_cache[:, None]

#         # =========================================================
#         # 2. 深度感知平滑随机丢弃 (Soft Mask)
#         # =========================================================
#         # A. 计算相机坐标系深度
#         ones = torch.ones((num_points, 1), device="cuda")
#         gaussian_positions_homo = torch.cat([means3D, ones], dim=1)
#         camera_coordinates = torch.matmul(gaussian_positions_homo, viewpoint_camera.world_view_transform.T)
#         camera_depths = camera_coordinates[:, 2] # Z-depth

#         # B. 平滑深度得分：取代硬性的 33%/67% 切分
#         # 使用指数衰减，让丢弃概率随距离增加平滑下降，减少训练抖动
#         d_min, d_max = camera_depths.min(), camera_depths.max()
#         depth_norm = (camera_depths - d_min) / (d_max - d_min + 1e-6)
#         smooth_depth_score = torch.exp(-2.0 * depth_norm) # 近处 1.0，远处趋近 0

#         # C. 密度归一化
#         density_norm = torch.ones_like(smooth_depth_score) * 0.5
#         if hasattr(pc, 'density_score'):
#             d_score = pc.density_score[:num_points]
#             density_norm = 1.0 - (d_score - d_score.min()) / (d_score.max() - d_score.min() + 1e-6)

#         # D. 综合计算丢弃概率与能量补偿
#         # 参数建议：depth_weight=0.5, density_weight=0.5
#         combined_score = (depth_weight * smooth_depth_score + density_weight * density_norm)
#         progress = min(1.0, iteration / 10000.0)
#         current_max_drop = drop_min + (drop_max - drop_min) * progress

#         # 限制最大丢弃率，防止梯度消失
#         drop_prob = torch.clamp(combined_score * current_max_drop, 0, 0.5)
#         keep_prob = 1.0 - drop_prob

#         # 生成随机掩码
#         random_mask = (torch.rand_like(keep_prob) < keep_prob).float()

#         # 【核心修改】执行 Dropout 补偿：opacity / keep_prob
#         # 这确保了渲染图像的整体亮度期望在训练和测试时保持一致
#         opacity = opacity * (random_mask / (keep_prob + 1e-7))[:, None]
    # =================================================================================

#     # Rasterize visible Gaussians to image
#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     rendered_image = rendered_image.clamp(0, 1)
#     out = {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter" : (radii > 0).nonzero(),
#         "radii": radii
#     }
    
#     return out


12.25# import torch
# import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh
# # 引入 knn 模块
# from torch_cluster import knn

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
#            override_color = None, is_train=False, iteration=None):
#     """
#     Render the scene. 
#     """
 
#     # Create zero tensor.
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug,
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # If precomputed 3d covariance is provided, use it.
#     scales = None
#     rotations = None
#     cov3D_precomp = None

#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     # SH -> RGB conversion
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # ==============================================================================
#     #  混合策略：KNN 结果复用逻辑
#     #  IF 部分: 计算并缓存 KNN 结构 + 密度消除
#     #  ELSE 部分: 复用 KNN 结构 + 随机消除
#     # ==============================================================================
#     if is_train and iteration is not None:
#         num_points = opacity.shape[0]

#         # 1. 初始化缓存
#         if not hasattr(pc, "density_mask_cache") or pc.density_mask_cache.shape[0] != num_points:
#             pc.density_mask_cache = torch.ones(num_points, dtype=torch.float32, device="cuda")
        
#         # 初始化 KNN 查表缓存 (用于 else 部分复用)
#         if not hasattr(pc, "knn_map_cache"):
#             pc.knn_map_cache = None

#         # --- [Phase A] 每 100 次迭代：重算 KNN，建立缓存，执行密度消除 ---
#         if iteration % 100 == 0:
#             k_density = 11  # 1个点 + 10个邻居
#             k_drop = 10     # 要消除周围的 10 个点

#             # 为了节省显存，使用 detach 的坐标
#             curr_xyz = means3D.detach()
            
#             # 1. 运行耗时的 KNN 计算
#             assign_idx = knn(curr_xyz, curr_xyz, k=k_density)
            
#             # 2. 【核心】保存 KNN 结果到缓存，供 else 复用
#             # 将 [2, N*k] 转为 [N, k] 的查表矩阵
#             pc.knn_map_cache = assign_idx[1].view(num_points, k_density)
            
#             # 3. 计算密度 (用于本次的密度消除)
#             # 利用刚算出的索引获取邻居坐标
#             neighbor_xyz = curr_xyz[pc.knn_map_cache] 
#             # 计算距离: [N, k]
#             sq_dist = ((curr_xyz.unsqueeze(1) - neighbor_xyz) ** 2).sum(dim=-1)
#             # 平均距离 (越小越密)
#             mean_dist = sq_dist[:, 1:].mean(dim=1) 

#             # 4. 筛选 Top 2% 最密集的点 (Density Selection)
#             num_seeds = int(num_points * 0.02 * min(1.0, iteration/10000.0))
            
#             if num_seeds > 0:
#                 # 选密度最大的
#                 sorted_indices = torch.argsort(mean_dist) 
#                 candidate_seeds = sorted_indices[:num_seeds]
                
#                 # 查表获取要消除的邻居
#                 neighbors_to_drop = pc.knn_map_cache[candidate_seeds].flatten()
                
#                 # 更新遮罩
#                 unique_drop_indices = torch.unique(neighbors_to_drop)
#                 pc.density_mask_cache.fill_(1.0)
#                 pc.density_mask_cache[unique_drop_indices] = 0.0

#             # 应用基于密度的遮罩
#             compensation = pc.density_mask_cache

#         # --- [Phase B] 其他迭代：复用 KNN 缓存 + 随机消除 ---
#         else:
#             # 1. 检查缓存是否可用 (防止点数变化导致索引越界)
#             if (pc.knn_map_cache is not None) and (pc.knn_map_cache.shape[0] == num_points):
                
#                 # 2. 计算需要多少个随机种子 (Random Selection)
#                 # drop_rate 随训练增加
#                 current_drop_rate = 0.05 * (iteration / 10000.0)
#                 # 除以 11 是因为每个种子会带走自己和10个邻居
#                 num_seeds = int((num_points * current_drop_rate) / 11)
                
#                 if num_seeds > 0:
#                     # 3. 随机选取种子点
#                     random_seeds = torch.randint(0, num_points, (num_seeds,), device="cuda")
                    
#                     # 4. 【复用】查表获取邻居 (无需运行 KNN)
#                     neighbors_to_drop = pc.knn_map_cache[random_seeds].flatten()
                    
#                     # 5. 制作临时遮罩
#                     unique_drop = torch.unique(neighbors_to_drop)
#                     compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                     compensation[unique_drop] = 0.0
                    
#                     # 6. 能量补偿 (Rescaling)
#                     # 因为是随机 dropout，为了保持画面亮度一致，需要放大剩余点的 opacity
#                     keep_ratio = 1.0 - (len(unique_drop) / num_points)
#                     compensation = compensation / (keep_ratio + 1e-7)
#                 else:
#                     compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")
#             else:
#                 # 缓存失效时的兜底
#                 compensation = torch.ones(num_points, dtype=torch.float32, device="cuda")

#         # 应用最终计算出的 compensation (来自 if 或 else)
#         opacity = opacity * compensation[:, None]

#     # ==============================================================================

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     rendered_image = rendered_image.clamp(0, 1)
#     out = {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter" : (radii > 0),
#         "radii": radii
#     }
    
#     return out


import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from torch_cluster import knn
#import rerun as rr


# 引入高效 KNN 库
try:
    from torch_cluster import knn
except ImportError:
    print("Please install torch_cluster: pip install torch-cluster")
    
#rr.init("3DGS_Debug", spawn=True)
# def visualize_drop_and_densify(iteration, means3D, mask, new_points=None):
#     """
#     iteration: 当前迭代次数
#     means3D: 当前所有点的坐标 [N, 3] (Tensor)
#     mask: 剔除掩码 [N], 0代表剔除, 1代表保留 (Tensor)
#     new_points: (可选) 本轮增密新产生的点 [M, 3] (Tensor)
#     """
    
#     # 1. 数据准备：转为 numpy
#     points_np = means3D.detach().cpu().numpy()
#     mask_np = mask.detach().cpu().numpy().astype(bool)
    
#     # 2. 分类点
#     # (A) 被保留的点 (Kept) -> 显示为灰色，半透明
#     kept_points = points_np[mask_np]
    
#     # (B) 被剔除的点 (Dropped) -> 显示为红色，醒目
#     dropped_points = points_np[~mask_np]
    
#     rr.set_time_sequence("step", iteration)
    
#     # --- 可视化层 1: 保留的点 (背景) ---
#     if len(kept_points) > 0:
#         rr.log(
#             "world/points_kept",
#             rr.Points3D(kept_points, colors=[200, 200, 200], radii=0.01)
#         )
        
#     # --- 可视化层 2: 被杀死的点 (红色) ---
#     if len(dropped_points) > 0:
#         rr.log(
#             "world/points_dropped",
#             rr.Points3D(dropped_points, colors=[255, 0, 0], radii=0.03) # 半径大一点，显眼
#         )

#     # --- 可视化层 3: 新增密/分裂的点 (绿色) ---
#     if new_points is not None:
#         new_np = new_points.detach().cpu().numpy()
#         if len(new_np) > 0:
#             rr.log(
#                 "world/points_born",
#                 rr.Points3D(new_np, colors=[0, 255, 0], radii=0.02)
#             )

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
           override_color = None, is_train=False, iteration=None):
    """
    Render the scene with structured KNN Dropout.
    """
 
    # 创建屏幕空间点的零张量，用于记录梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    num_points = opacity.shape[0]

    # 获取协方差参数
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 处理颜色 (SH 转换)
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

#     # ==============================================================================
#     # 核心逻辑：周期性 KNN 重算 + 每一帧查表随机 Dropout
#     # ==============================================================================
# #     if is_train and iteration is not None:
# #             num_points = opacity.shape[0]
# #             # 种子比例：随迭代增加
# #             seed_ratio = 0.02 * (iteration / 10000) 
# #             # 如果你想丢弃种子+周围10个点，k_total 应为 11
# #             k_total = 10  

# #             # 1. 检查并初始化缓存
# #             if not hasattr(pc, "knn_map_cache") or pc.knn_map_cache is None or pc.knn_map_cache.shape[0] != num_points:
# #                 pc.knn_map_cache = None

# #             # --- A. 周期性更新：100的倍数时重算全量 KNN ---
# #             if iteration % 100 == 0:
# #                 curr_xyz = means3D.detach()
# #                 # 计算全量 KNN (只在这一步耗时)
# #                 assign_idx = knn(curr_xyz, curr_xyz, k=k_total)
# #                 # 建立索引映射表 [N, k_total]
# #                 pc.knn_map_cache = assign_idx[1].view(num_points, k_total)

# #             # --- B. 执行 Drop 逻辑：无论是重算帧还是复用帧都会运行到这里 ---
# #             if pc.knn_map_cache is not None:
# #                 # 1. 确定种子数量
# #                 num_seeds = max(1, int(num_points * seed_ratio))
# #                 num_seeds = min(num_seeds, num_points)

# #                 # 2. 随机选择种子索引
# #                 seed_indices = torch.randint(0, num_points, (num_seeds,), device="cuda")

# #                 # 3. 查表获取种子点及其邻居的索引 (Flatten 后包含种子和其所有邻居)
# #                 raw_drop_indices = pc.knn_map_cache[seed_indices].flatten()

# #                 # 4. 去重处理 (防止不同种子的邻居重叠导致的重复计算)
# #                 unique_drop_indices = torch.unique(raw_drop_indices)

# #                 # 5. 构造 Mask
# #                 mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
# #                 mask[unique_drop_indices] = 0.0

# #                 # 6. 计算实际存活率并进行能量补偿
# #                 actual_keep_count = num_points - unique_drop_indices.numel()
# #                 current_keep_ratio = actual_keep_count / num_points

# #                 # 补偿因子：1 / 存活率，确保画面亮度期望值不变
# #                 rescale_factor = 1.0 / (max(current_keep_ratio, 0.1) + 1e-7)

# #                 # 同时应用 Mask 和补偿
# #                 opacity = opacity * (mask * rescale_factor)[:, None]
            
# #     # ==============================================================================
#     if is_train :#and iteration is not None:
#             num_points = opacity.shape[0]

#             # 1. 初始化缓存
#             if not hasattr(pc, "density_mask_cache") or pc.density_mask_cache.shape[0] != num_points:
#                 pc.density_mask_cache = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                 pc.knn_neigh_idx = None # 存储邻居索引表 [N, 11]

#             # 每 100 次迭代更新一次全局邻居结构
#             if iteration % 100 == 0:
#                 k_total = 11  # 种子(1) + 邻居(10)
#                 print("KNN")
#                 curr_xyz = means3D.detach()

#                 # --- A. 全局计算 KNN 并缓存索引表 ---
#                 assign_idx = knn(curr_xyz, curr_xyz, k=k_total)
#                 # 建立查表矩阵 [N, 11]，pc.knn_neigh_idx[i] 就是点 i 及其 10 个邻居
#                 pc.knn_neigh_idx = assign_idx[1].view(num_points, k_total)

#                 # --- B. 计算密度并执行“密度丢弃” ---
#                 diff = curr_xyz[assign_idx[0]] - curr_xyz[assign_idx[1]]
#                 sq_dist = (diff ** 2).sum(dim=1).view(num_points, k_total)
#                 mean_dist = sq_dist[:, 1:].mean(dim=1) 

#                 sorted_indices = torch.argsort(mean_dist) 
#                 num_seeds = int(num_points * 0.02 * (iteration / 10000))
#                 candidate_seeds = sorted_indices[:num_seeds]

#                 # 查表获取邻居并去重
#                 drop_indices = pc.knn_neigh_idx[candidate_seeds].flatten()
#                 unique_drop = torch.unique(drop_indices)

#                 pc.density_mask_cache.fill_(1.0)
#                 pc.density_mask_cache[unique_drop] = 0.0

#                 # 应用遮罩
#                 opacity = opacity * pc.density_mask_cache[:, None]

#             else:
#                 # --- C. 非计算帧：复用 KNN 结构执行“随机丢弃” ---
#                 # 只有当缓存存在且点数匹配时执行
#                 if  pc.knn_neigh_idx.shape[0] == num_points:
#                 #if pc.knn_neigh_idx is not None and pc.knn_neigh_idx.shape[0] == num_points:
#                     # 计算当前需要的随机种子数量
#                     drop_rate = 0.1 * (iteration / 10000)
#                     if (iteration%10)==0:
#                         print("dropping")
#                     # 预估种子数：(总点数 * 丢弃率) / 每个簇的大小(11)
#                     num_seeds = int((num_points * drop_rate) / 11)

#                     if num_seeds > 0:
#                         # 1. 随机选择种子点索引
#                         rand_seeds = torch.randint(0, num_points, (num_seeds,), device="cuda")

#                         # 2. 【复用查表】直接获取这些种子周围的 10 个邻居
#                         # 这一步极快，完全没有空间搜索计算
#                         drop_indices = pc.knn_neigh_idx[rand_seeds].flatten()
#                         unique_drop = torch.unique(drop_indices)

#                         # 3. 生成临时遮罩并执行能量补偿
#                         temp_mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                         temp_mask[unique_drop] = 0.0

#                         keep_ratio = 1.0 - (unique_drop.numel() / num_points)
#                         # 应用遮罩并进行亮度补偿
#                         opacity = opacity * (temp_mask / (keep_ratio + 1e-7))[:, None]
#                 else:
#                     # 兜底：如果点数变了导致缓存失效，则不执行丢弃或执行普通随机 Dropout
#                     if (iteration//10)==0:
#                         print("d")
#                     compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
#                      # Apply DropGaussian with compensation
#                     drop_rate = 0.2 * (iteration/10000)
#                     d = torch.nn.Dropout(p=drop_rate)
#                     # Apply to opacity
#                     opacity = opacity * compensation[:, None]
#                     compensation = d(compensation) 
#12-28
#     if is_train and iteration is not None:
#             # --- A. 核心同步检查 ---
#             # 触发条件：1.无缓存 2.100轮周期 3.检测到增密/剪枝导致点数变化
#             needs_knn_update = (not hasattr(pc, "knn_neigh_idx") or 
#                                 pc.knn_neigh_idx is None or 
#                                 iteration % 100 == 0 or 
#                                 pc.knn_neigh_idx.shape[0] != num_points)

#             if needs_knn_update:
#                 k_total = 22  # 1个点 + 10个邻居#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                 #if iteration % 100 == 0:
#                  #   print(f"[KNN Update] Iteration {iteration}: Points changed to {num_points}. Recalculating neighbors...")

#                 curr_xyz = means3D.detach()
#                 # 计算全量 KNN 并建立查表缓存 [N, 11]
#                 assign_idx = knn(curr_xyz, curr_xyz, k=k_total)
#                 pc.knn_neigh_idx = assign_idx[1].view(num_points, k_total)

#                 # 执行基于密度的初始丢弃 (Phase 1)
#                 diff = curr_xyz[assign_idx[0]] - curr_xyz[assign_idx[1]]
#                 sq_dist = (diff ** 2).sum(dim=1).view(num_points, k_total)
#                 mean_dist = sq_dist[:, 1:].mean(dim=1) 

#                 sorted_indices = torch.argsort(mean_dist) 
#                 num_seeds = int(num_points * 0.25 * min(1.0, iteration / 10000))#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                 candidate_seeds = sorted_indices[:num_seeds]

#                 # 结构化置零
#                 drop_indices = pc.knn_neigh_idx[candidate_seeds].flatten()
#                 unique_drop = torch.unique(drop_indices)

#                 mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                 mask[unique_drop] = 0.0
#                 opacity = opacity * mask[:, None]

#             else:
#                 # --- B. 非重算帧：复用 KNN 结构执行“随机结构化丢弃” ---
#                 # 这里的 pc.knn_neigh_idx.shape[0] 必然等于 num_points
#                 drop_rate = 0.25 * (iteration / 10000)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                 num_seeds = int((num_points * drop_rate) / 22)#!!!!!!!!!!!!!!!!!!!!!!!!!

#                 if num_seeds > 0:
#                     # 随机选择种子中心
#                     rand_seeds = torch.randint(0, num_points, (num_seeds,), device="cuda")
#                     # 【复用查表】直接获取邻居索引
#                     drop_indices = pc.knn_neigh_idx[rand_seeds].flatten()
#                     unique_drop = torch.unique(drop_indices)

#                     # 生成掩码并计算亮度补偿系数
#                     temp_mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                     temp_mask[unique_drop] = 0.0

#                     # 能量补偿：存活率 keep_ratio = 存活点数 / 总点数
#                     keep_ratio = 1.0 - (unique_drop.numel() / num_points)
#                     rescale_factor = 1.0 / (keep_ratio + 1e-7)

#                     # 应用遮罩与补偿
#                     opacity = opacity * (temp_mask * rescale_factor)[:, None]

#                     #if iteration % 100 == 10: # 仅在更新后的几帧打印，避免刷屏
#                      #   print(f"[Dropout Active] Iteration {iteration}: Dropping clusters based on {num_seeds} seeds.")
#                 else:
#                     # 若 drop_rate 太小，种子数为0，则不做任何操作
#                     pass
#     if is_train and iteration is not None:

#             # --- [用户参数配置区] ---
#             # Z: 连带丢弃的邻居数量 (k_total = 1 + Z)
#             Z_neighbors = 20 
#             k_total = 1 + Z_neighbors 

#             # X: 定义"密集区域"的比例 (前 X% 最密集的点进入候选池)
#             X_percent = 0.01 

#             # Y: 基础丢弃概率 (最终丢弃率 = Y * progress)
#             Y_base_rate = 0.25 
#             # ------------------------

#             # 1. 检查并初始化缓存容器
#             if not hasattr(pc, "knn_neigh_idx") or pc.knn_neigh_idx is None or pc.knn_neigh_idx.shape[0] != num_points:
#                 pc.knn_neigh_idx = None # 存储 [N, k_total] 的邻居表
#                 pc.dense_pool_cache = None # 存储前 X% 密集点的索引

#             # --- Phase A: 核心同步检查与 KNN 更新 ---
#             # 触发条件：1.缓存不存在 2.每100次迭代 3.点数发生物理变化(增密/剪枝)
#             needs_knn_update = (pc.knn_neigh_idx is None or 
#                                 iteration % 100 == 0 or 
#                                 pc.knn_neigh_idx.shape[0] != num_points)

#             if needs_knn_update:
#                 # [标记 1] 打印 KNN 更新状态
                

#                 curr_xyz = means3D.detach()

#                 # 1. 全局计算 KNN
#                 assign_idx = knn(curr_xyz, curr_xyz, k=k_total)

#                 # 2. 建立并缓存邻居查表矩阵 [N, 1 + Z]
#                 pc.knn_neigh_idx = assign_idx[1].view(num_points, k_total)

#                 # 3. 计算所有点的密集程度 (Mean Squared Distance to Neighbors)
#                 # 距离越小，代表越密集
#                 diff = curr_xyz[assign_idx[0]] - curr_xyz[assign_idx[1]]
#                 sq_dist = (diff ** 2).sum(dim=1).view(num_points, k_total)
#                 mean_dist = sq_dist[:, 1:].mean(dim=1) 

#                 # 4. 排序并筛选前 X% 密集点作为"密集池"
#                 sorted_indices = torch.argsort(mean_dist) # 从小到大排序 (密 -> 疏)
#                 num_pool = int(num_points * X_percent)
#                 # 缓存这个池子，供后续复用
#                 pc.dense_pool_cache = sorted_indices[:num_pool]

#             # --- Phase B: 执行 Drop 逻辑 (更新帧和复用帧通用) ---
#             # 只要缓存有效，就执行丢弃
#             if pc.dense_pool_cache is not None and pc.knn_neigh_idx is not None:

#                 # 1. 确定本轮要丢弃的种子数量 (基于密集池的大小 * Y%)
#                 # 随迭代增加强度
#                 current_Y_rate = Y_base_rate * min(1.0, iteration / 10000.0)
#                 num_seeds = int(pc.dense_pool_cache.numel() * current_Y_rate)

#                 if num_seeds > 0:
#                     # 2. 从【密集池】中随机选择种子
#                     # 注意：这里是在 dense_pool_cache 里抽样，不是全图抽样
#                     rand_perm = torch.randperm(pc.dense_pool_cache.numel(), device="cuda")[:num_seeds]
#                     selected_seed_indices = pc.dense_pool_cache[rand_perm]

#                     # 3. 【查表连带】获取种子点及其 Z 个邻居
#                     # 这一步复用了 Phase A 算好的表
#                     drop_indices = pc.knn_neigh_idx[selected_seed_indices].flatten()

#                     # 4. 去重
#                     unique_drop = torch.unique(drop_indices)

#                     # 5. 生成 Mask
#                     mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                     mask[unique_drop] = 0.0

#                     # 6. 能量补偿 (保持画面平均亮度)
#                     keep_ratio = 1.0 - (unique_drop.numel() / num_points)
#                     rescale_factor = 1.0 / (keep_ratio + 1e-7)

#                     # 应用
#                     opacity = opacity * (mask * rescale_factor)[:, None]

#                     # [标记 2] 打印 Drop 执行状态 (每100帧或者特定帧打印，防止刷屏)
                   
                        
#             else:
#                 # 兜底：缓存异常时不操作
#                 pass
#体素+KNN
#     if is_train and iteration is not None:
            
#             num_points = opacity.shape[0]
#             #expected_resolution = math.pow(num_points, 1.0/3.0)
#             #voxel_ratio = 1.0 / (expected_resolution * 2.0 + 1e-6)
#             #voxel_ratio = max(0.002, min(0.02, voxel_ratio))
#             #Z_neighbors = int(10 * (num_points / 10000.0))
#             #Z_neighbors = min(Z_neighbors, 15)

#             # --- [1. 参数配置] ---
#             Y_dense_percent = 0.20
#             voxel_ratio = 0.00001*iteration    # 场景跨度的比例
#             Z_neighbors = int(5*(iteration/10000))    # 连带邻居数

#             # --- [2. KNN 密度排名更新 (定期执行) ---
#             # 仅在每 100 步或缓存丢失时更新精确的密度排名
#             needs_knn_update = (not hasattr(pc, "sorted_dense_indices") or 
#                                 pc.sorted_dense_indices is None or 
#                                 iteration % 100 == 0)

#             if needs_knn_update:
                
#                 curr_xyz_knn = means3D.detach().contiguous()
#                 # 计算密度用的邻居数
#                 k_knn =  int(10 *(iteration/10000))
#                 assign_idx = knn(curr_xyz_knn, curr_xyz_knn, k=k_knn+1)
#                 diff = curr_xyz_knn[assign_idx[0]] - curr_xyz_knn[assign_idx[1]]
#                 mean_dist = (diff ** 2).sum(dim=-1).reshape(num_points, k_knn+1)[:, 1:].mean(dim=1)

#                 # 缓存：当前点云中最密集的 Top 索引排名
#                 pc.sorted_dense_indices = torch.argsort(mean_dist)
#                 pc.knn_update_num = num_points # 记录更新时的点数
#                 #print(num_points)
#                 # 缓存邻居表 (用于 KNN 环节的连带丢弃)
#                 pc.knn_neigh_idx = assign_idx[1].reshape(num_points, k_knn+1)

#             # --- [3. 动态体素映射 (每一帧执行) ---
#             curr_xyz = means3D.detach()
#             # 动态计算体素大小以适配场景
#             scene_extent = (curr_xyz.max(dim=0)[0] - curr_xyz.min(dim=0)[0]).max()
#             dynamic_voxel_size = scene_extent * voxel_ratio

#             # 将当前点投射到体素
#             voxel_coords = (curr_xyz / dynamic_voxel_size).int()
#             unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
#             num_voxels = unique_voxels.shape[0]

#             # --- [4. KNN-体素-KNN 逻辑决策] ---
#             current_ratio = Y_dense_percent * min(1.0, iteration / 10000.0)
#             num_seeds = int(num_points * current_ratio)

#             mask = torch.ones(num_points, dtype=torch.float32, device="cuda")

#             if num_seeds > 0:
#                 # 判定：如果当前点数与 KNN 缓存点数一致，使用“索引+邻居”丢弃
#                 if pc.knn_update_num == num_points:
#                     # 模式 A: 纯索引 KNN 连带丢弃 (最精确)
#                     seeds = pc.sorted_dense_indices[:num_seeds]
#                     # 连带邻居
#                     drop_idx = pc.knn_neigh_idx[seeds, :Z_neighbors+1].flatten()
#                     unique_drop = torch.unique(drop_idx[drop_idx < num_points])
#                     mask[unique_drop] = 0.8
#                     mask=(torch.rand_like(mask)<mask).float()
#                     #if iteration % 10 == 0:
#                      #   print(f"\033[1;33m[Normal dropping]\033[0m Points: {num_points}")
                    
#                 else:
#                     # 模式 B: 体素传播丢弃 (增密期间的鲁棒模式)
#                     # 1. 找到“种子索引”在当前体素中的位置
#                     # 即使点数变了，pc.sorted_dense_indices 的前部依然代表了那些较老的、密集的区域
#                     valid_seed_limit = min(num_seeds, pc.sorted_dense_indices.shape[0])
#                     safe_seeds = pc.sorted_dense_indices[:valid_seed_limit]
#                     safe_seeds = safe_seeds[safe_seeds < num_points] # 再次越界保护

#                     # 2. 找到这些种子点落在了哪些体素里
#                     drop_voxel_ids = inverse_indices[safe_seeds]

#                     # 3. 标记这些体素为“待删除簇”
#                     voxel_drop_flags = torch.ones(num_voxels, dtype=torch.float32, device="cuda")
#                     voxel_drop_flags[drop_voxel_ids] = 0.8

#                     # 4. 删掉属于这些密集型体素的所有点 (实现非随机成簇)
#                     mask = (voxel_drop_flags[inverse_indices]).float()
#                     mask=(torch.rand_like(mask)<mask).float()
#                     #visualize_drop_and_densify(iteration, means3D, mask)
#                     #if iteration % 10 == 0:
#                      #   print(f"\033[1;33m[VOXEL-MAPPING]\033[0m Points: {num_points} | Propagating dense seeds via Voxels.")

#             # --- [5. 补偿与应用] ---
#             actual_keep_ratio = mask.mean()
#             if actual_keep_ratio < 1.0:
#                 rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#                 opacity = opacity * (mask * rescale_factor)[:, None]
#体素选前X%drop    if is_train and iteration is not None:
        # --- [1. 参数配置] ---
        # 动态调整体素大小，随迭代次数增加，关注更精细的区域
#         voxel_ratio = 0.00001 * iteration #+ 0.0001 

#         # 设定“密集”的阈值：如果一个体素内的点超过这个数，视为密集区域
#         # 这个阈值可以随时间衰减，或者固定。这里举例为根据点云总数动态调整
#         curr_xyz = means3D.detach()
#         num_points = curr_xyz.shape[0]

#         # --- [2. 实时体素密度计算 (替代 KNN)] ---
#         # 计算场景范围
#         scene_extent = (curr_xyz.max(dim=0)[0] - curr_xyz.min(dim=0)[0]).max()
#         dynamic_voxel_size = scene_extent * voxel_ratio

#         # 投射到整数体素坐标
#         voxel_coords = (curr_xyz / dynamic_voxel_size).int()

#         # 这里使用 unique 的 return_counts 功能直接获得密度！
#         # inverse_indices: 每个点对应 unique_voxels 中的哪个索引
#         # counts: 每个 unique voxel 里有多少个点
#         unique_voxels, inverse_indices, counts = torch.unique(
#             voxel_coords, dim=0, return_inverse=True, return_counts=True
#         )

#         # --- [3. 识别密集区域] ---
#         # 获得每个点所在体素的“拥挤程度”
#         point_densities = counts[inverse_indices] # shape: [num_points]

#         # 定义高密度阈值，例如：如果该点周围体素点数超过 90% 分位数的体素，则视为密集
#         # 或者简单粗暴：如果体素内点数 > N (例如 20)
#         density_threshold = torch.quantile(point_densities.float(), 0.7) # 动态阈值：只处理最密集的 10% 区域

#         # 生成 Mask
#         # 逻辑：只有当点处于高密度区域时，才有可能被 Drop
#         is_dense_mask = point_densities > density_threshold

#         # --- [4. 执行 Dropout] ---
#         if is_dense_mask.any():
#             # 创建基础 mask (全 1)
#             mask = torch.ones(num_points, dtype=torch.float32, device="cuda")

#             # 定义 Dropout 概率，例如 0.5 (丢弃 50%)
#             # 只针对密集区域生成随机 mask
#             drop_prob = 0.5
#             random_tensor = torch.rand(num_points, device="cuda")

#             # 逻辑：如果是密集点 AND 随机数 < 概率 -> 置为 0 (Drop)
#             # 也就是：保留 mask = 1, 除非 (is_dense AND rand < drop_prob)
#             drop_condition = torch.logical_and(is_dense_mask, random_tensor < drop_prob)
#             mask[drop_condition] = 0.0
#             actual_keep_ratio = mask.mean()
#             if actual_keep_ratio < 1.0:
#                 rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#                 opacity = opacity * (mask * rescale_factor)[:, None]
            #opacity = opacity * mask[:, None]
            
            #if iteration % 100 == 0 and pc.knn_update_num == num_points:
             #   print(f"\033[1;32m[KNN-DIRECT]\033[0m Precise Index Drop Active.",num_points)

                
                
#遍历随机选体素    if is_train and iteration is not None:
        # --- [1. 参数配置] ---
        # 动态调整体素大小，随迭代次数增加，关注更精细的区域
#         voxel_ratio = 0.0000001 * iteration #+ 0.0001 

#         # 核心参数：随机选择体素进行丢弃的概率（比如选30%的体素丢弃）
#         voxel_drop_prob = 0.1  # 可根据需求调整，0.3表示随机丢弃30%的体素

#         # --- [2. 实时体素划分] ---
#         curr_xyz = means3D.detach()
#         num_points = curr_xyz.shape[0]

#         # 计算场景范围
#         scene_extent = (curr_xyz.max(dim=0)[0] - curr_xyz.min(dim=0)[0]).max()
#         dynamic_voxel_size = scene_extent * voxel_ratio

#         # 投射到整数体素坐标
#         voxel_coords = (curr_xyz / dynamic_voxel_size).int()

#         # 获取唯一体素、每个点对应的体素索引、每个体素的点数
#         unique_voxels, inverse_indices, counts = torch.unique(
#             voxel_coords, dim=0, return_inverse=True, return_counts=True
#         )
#         num_unique_voxels = len(unique_voxels)

#         # --- [3. 核心逻辑：随机选体素并丢弃其所有点] ---
#         # 创建基础mask（全1，表示默认保留所有点）
#         mask = torch.ones(num_points, dtype=torch.float32, device="cuda")

#         # 步骤1：生成随机数，筛选出要丢弃的体素索引
#         # 为每个唯一体素生成一个0~1的随机数，小于voxel_drop_prob的体素被选中丢弃
#         voxel_random = torch.rand(num_unique_voxels, device="cuda")
#         dropped_voxel_indices = torch.where(voxel_random < voxel_drop_prob)[0]  # 被丢弃的体素索引

#         if len(dropped_voxel_indices) > 0:
#             # 步骤2：找到这些被丢弃体素对应的所有点
#             # 遍历每个被丢弃的体素索引，标记其所有点为丢弃
#             for voxel_idx in dropped_voxel_indices:
#                 point_indices_in_voxel = (inverse_indices == voxel_idx)
#                 mask[point_indices_in_voxel] = 0.0  # 整个体素的点全部置为0（丢弃）

#         # --- [4. 比例校准与Opacity调整（保留原有逻辑）] ---
#         actual_keep_ratio = mask.mean()
#         if actual_keep_ratio < 1.0:
#             rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#             opacity = opacity * (mask * rescale_factor)[:, None]
#随机体素选择 
#     if is_train and iteration is not None:
#                 # --- [1. 参数配置] ---
#                 voxel_ratio = 0.003
#                 curr_xyz = means3D.detach()
#                 num_points = curr_xyz.shape[0]

#                 # --- [2. 空间体素化与索引获取] ---
#                 scene_extent = (curr_xyz.max(dim=0)[0] - curr_xyz.min(dim=0)[0]).max()
#                 dynamic_voxel_size = scene_extent * voxel_ratio
#                 voxel_coords = (curr_xyz / dynamic_voxel_size).int()

#                 # unique_voxels: 唯一体素的坐标
#                 # inverse_indices: 每个点所属体素在 unique_voxels 中的索引
#                 unique_voxels, inverse_indices = torch.unique(
#                     voxel_coords, dim=0, return_inverse=True
#                 )

#                 num_unique_voxels = unique_voxels.shape[0]

#                 # --- [3. 随机挑选 X% 的体素] ---
#                 # 设定你想挑选的体素比例，例如随机选 30% 的体素进行操作
#                 voxel_selection_ratio = 0.25*(iteration/10000)
#                 # 为每个“唯一体素”生成一个随机数
#                 voxel_random_vals = torch.rand(num_unique_voxels, device="cuda")

#                 # 确定哪些体素被选中（布尔 Mask，长度等于唯一体素的数量）
#                 selected_voxels_mask = voxel_random_vals < voxel_selection_ratio
#                 counts = torch.bincount(inverse_indices)
#                 # 仅计算被选中体素的点数均值
#                 if selected_voxels_mask.any():
#                     selected_counts = counts[selected_voxels_mask]
#                     avg_points_in_selected = selected_counts.float().mean().item()
#                     print(f"Average points in SELECTED voxels: {avg_points_in_selected:.2f}")
#                 # --- [4. 执行 Drop 操作] ---
#                 # 将“体素级掩码”映射回“点级掩码”
#                 # 这样我们就知道哪些点属于被选中的那 X% 的体素
#                 is_in_selected_voxel = selected_voxels_mask[inverse_indices]

#                 # 定义在被选中的体素内部的 Drop 概率（例如在这些体素里丢掉 50% 的点）
#                 # 如果你想把选中的体素整个删掉，就把 inner_drop_prob 设为 1.0
#                 inner_drop_prob = 1.0
#                 random_tensor = torch.rand(num_points, device="cuda")

#                 # 最终 Drop 条件：点属于被选中的体素，且随机数命中 Drop 概率
#                 drop_condition = torch.logical_and(is_in_selected_voxel, random_tensor < inner_drop_prob)

#                 mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                 mask[drop_condition] = 0.0
#                 # --- [5. 补偿与应用] ---
#                 actual_keep_ratio = mask.mean()
#                 if actual_keep_ratio < 1.0:
#                     rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#                     opacity = opacity * (mask * rescale_factor)[:, None]

# DropGaussian
#     if is_train:
#         # Create initial compensation factor (1 for each Gaussian)
#         compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

#         # Apply DropGaussian with compensation
#         drop_rate = 0.2 * (iteration/10000)
#         d = torch.nn.Dropout(p=drop_rate)
#         compensation = d(compensation)
#         # Apply to opacity
#         opacity = opacity * compensation[:, None]
# train.py中修改逻辑：先选点，再生成compensation
    # if is_train:
    #     # 1. 先调用SH选点逻辑，获取选中索引（需修改方法返回选中索引）
    #     selected_indices = None
    #     if iteration <= 6000: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #         selected_indices = pc.restrict_sh_degree_by_points(iteration, random_pct=0.2*(iteration/10000))  
    #     #选点开启SH系数，0 1 2 3阶的系数，不要重叠的drop,对低阶的点选点进行drop
    #     #0-1000阶段开启0阶：
    #     # 2. 生成compensation张量，与SH选点对应
    #         compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
    #         if selected_indices is not None and len(selected_indices) > 0:
    #         # 让SH选中的点，对应compensation中的0（被Dropout丢弃）
    #             #print(selected_indices)
    #             # if 0<=iteration<2000:
    #             #     drop_prob = 0.2
    #             # elif 2000<=iteration<4000:
    #             #     drop_prob = 0.4
    #             # elif 4000<=iteration<=6000:
    #             drop_prob = 1.0
    #             num_second = int(len(selected_indices) * drop_prob)
    #         # 3. 从 selected_indices 中随机抽取 num_second 个索引（生成新索引）
    #         # 原理：先对 selected_indices 的长度做随机排列，再切片（等价于随机抽样）
    #             new_selected_indices = selected_indices[torch.randperm(len(selected_indices))[:num_second]]
    #             compensation[new_selected_indices] = 0.0
    #         # 3. 后续DropGaussian逻辑（可省略nn.Dropout，直接使用生成的compensation）
    #             drop_rate = 0.2 * (iteration/10000)
    #     # 若需保留Dropout的缩放特性，可补充缩放逻辑
    #             compensation = compensation * (1 / (1 - drop_rate))
    #             opacity = opacity * compensation[:, None]
    #             #print( num_second)
    #     else:
    #         #print(iteration)
    #         dcompensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
    #         drop_rate = 0.2 * (iteration/10000)
    #         d = torch.nn.Dropout(p=drop_rate)
    #         dcompensation = d(dcompensation)
    #         opacity = opacity * dcompensation[:, None]

        # # 4. Apply to opacity
        # opacity = opacity * compensation[:, None]
#     if is_train:
#             selected_indices = None
#             if iteration <= 5000: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                 selected_indices = pc.restrict_sh_degree_by_points(iteration, random_pct=0.25*(iteration/10000))#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
#             # --- A. 核心同步检查 ---
#             # 触发条件：1.无缓存 2.100轮周期 3.检测到增密/剪枝导致点数变化
#             needs_knn_update = (not hasattr(pc, "knn_neigh_idx") or 
#                                 pc.knn_neigh_idx is None or 
#                                 iteration % 100 == 0 or 
#                                 pc.knn_neigh_idx.shape[0] != num_points)

#             if needs_knn_update:
#                 k_total = 22  # 1个点 + 10个邻居#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                 #if iteration % 100 == 0:
#                  #   print(f"[KNN Update] Iteration {iteration}: Points changed to {num_points}. Recalculating neighbors...")

#                 curr_xyz = means3D.detach()
#                 # 计算全量 KNN 并建立查表缓存 [N, 11]
#                 assign_idx = knn(curr_xyz, curr_xyz, k=k_total)
#                 pc.knn_neigh_idx = assign_idx[1].view(num_points, k_total)

#                 # 执行基于密度的初始丢弃 (Phase 1)
#                 diff = curr_xyz[assign_idx[0]] - curr_xyz[assign_idx[1]]
#                 sq_dist = (diff ** 2).sum(dim=1).view(num_points, k_total)
#                 mean_dist = sq_dist[:, 1:].mean(dim=1) 

#                 sorted_indices = torch.argsort(mean_dist) 
#                 num_seeds = int(num_points * 0.2 * min(1.0, iteration / 10000))#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                 candidate_seeds = sorted_indices[:num_seeds]

#                 # 结构化置零
#                 drop_indices = pc.knn_neigh_idx[candidate_seeds].flatten()
#                 unique_drop = torch.unique(drop_indices)
#                 if selected_indices!=None:
#                     combined_tensor = torch.cat([selected_indices, unique_drop], dim=0)
#                     union = torch.unique(combined_tensor)
#                     mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                     mask[union] = 0.0
#                 else:
#                     mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                     mask[unique_drop] = 0.0
#                 opacity = opacity * mask[:, None]

#             else:
#                 # --- B. 非重算帧：复用 KNN 结构执行“随机结构化丢弃” ---
#                 # 这里的 pc.knn_neigh_idx.shape[0] 必然等于 num_points
#                 drop_rate = 0.25 * (iteration / 10000)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                 num_seeds = int((num_points * drop_rate) / 22)#!!!!!!!!!!!!!!!!!!!!!!!!!

#                 if num_seeds > 0:
#                     # 随机选择种子中心
#                     rand_seeds = torch.randint(0, num_points, (num_seeds,), device="cuda")
#                     # 【复用查表】直接获取邻居索引
#                     drop_indices = pc.knn_neigh_idx[rand_seeds].flatten()
#                     unique_drop = torch.unique(drop_indices)

#                     # 生成掩码并计算亮度补偿系数
#                     temp_mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                     if selected_indices!=None:
#                         combined_tensor = torch.cat([selected_indices, unique_drop], dim=0)
#                         union = torch.unique(combined_tensor)
#                         temp_mask[union] = 0.0
#                          # 能量补偿：存活率 keep_ratio = 存活点数 / 总点数
#                         keep_ratio = 1.0 - (union.numel() / num_points)
#                         rescale_factor = 1.0 / (keep_ratio + 1e-7)

#                     # 应用遮罩与补偿
#                         opacity = opacity * (temp_mask * rescale_factor)[:, None]
#                     else:
#                         temp_mask[unique_drop] = 0.0
#                         keep_ratio = 1.0 - (unique_drop.numel() / num_points)
#                         rescale_factor = 1.0 / (keep_ratio + 1e-7)
#                         opacity = opacity * (temp_mask * rescale_factor)[:, None]
#                     #if iteration % 100 == 10: # 仅在更新后的几帧打印，避免刷屏
#                      #   print(f"[Dropout Active] Iteration {iteration}: Dropping clusters based on {num_seeds} seeds.")
#                 else:
#                     # 若 drop_rate 太小，种子数为0，则不做任何操作
#                     pass
#1-21体素取并集
#     if is_train and iteration is not None:
#         # --- [1. 参数配置] ---
#         #import random
#         #ratio=random.uniform(0.001,0.003)
#         progress = iteration / 10000.0
    
        
#         voxel_ratio = 0.001 #*(1-progress)+ 0.0015 * progress
#         #voxel_ratio =0.001#*min(2*(iteration/10000),1)
#         curr_xyz = means3D#.detach()
#         num_points = curr_xyz.shape[0]
#         min_points_per_voxel =20   # 新增：定义体素点数阈值（≥20）

#         # --- [2. 空间体素化与索引获取] ---
#         q_max = torch.quantile(curr_xyz, 0.98, dim=0)#取98%分位数
#         q_min = torch.quantile(curr_xyz, 0.02, dim=0)
#         scene_extent = (q_max - q_min).max()#场景大小
#         #scene_extent = (curr_xyz.max(dim=0)[0] - curr_xyz.min(dim=0)[0]).max()
#         dynamic_voxel_size = scene_extent * voxel_ratio
#         voxel_coords = (curr_xyz / dynamic_voxel_size).int()#坐标/体素大小取整即划分体素

#         # unique_voxels: 唯一体素的坐标
#         # inverse_indices: 每个点所属体素在 unique_voxels 中的索引
#         unique_voxels, inverse_indices = torch.unique(
#             voxel_coords, dim=0, return_inverse=True
#         )

#         num_unique_voxels = unique_voxels.shape[0]#一维张量，体素的数目

#         # --- [3. 筛选满足点数阈值（≥20）且随机挑选 X% 的体素] ---
#         # 步骤3.1：统计每个唯一体素包含的点数
#         counts = torch.bincount(inverse_indices, minlength=num_unique_voxels)  # 确保长度与唯一体素一致
#         #返回非负整数的数量

#         # 步骤3.2：新增：生成「体素点数≥20」的掩码（布尔张量，长度=num_unique_voxels）
#         voxel_has_enough_points = (counts>=min_points_per_voxel)#>= min_points_per_voxel
#         #一维张量

#         # 步骤3.3：设定随机挑选体素的比例
#         voxel_selection_ratio = 0.25*(iteration/10000)
#         voxel_random_vals = torch.rand(num_unique_voxels, device="cuda")
#         #一维张量
        
#         # 步骤3.4：生成「随机选中体素」的掩码
#         voxel_randomly_selected = voxel_random_vals < voxel_selection_ratio
#         #一维张量
#         # 步骤3.5：组合两个掩码：仅保留「点数≥20 且 被随机选中」的体素（核心修改）
#         selected_voxels_mask = torch.logical_or(voxel_has_enough_points, voxel_randomly_selected)
#         #逻辑或，仍然是一维张量
#         # 步骤3.6：仅计算满足条件的体素的点数均值（优化打印）
#         if selected_voxels_mask.any():
#             selected_counts = counts[selected_voxels_mask]
#             avg_points_in_selected = selected_counts.float().mean().item()
#             #print(f"Average points in SELECTED voxels (≥20 points): {avg_points_in_selected:.2f}")
#         #else:
#             #print("No voxels meet the condition (≥20 points and randomly selected)")

#         # --- [4. 执行 Drop 操作] ---
#         # 将「体素级掩码」映射回「点级掩码」：哪些点属于满足条件的体素
#         is_in_selected_voxel = selected_voxels_mask[inverse_indices]

#         # 定义在被选中的体素内部的 Drop 概率（1.0 表示删掉体素内所有点）
#         inner_drop_prob = 1
#         random_tensor = torch.rand(num_points, device="cuda")

#         # 最终 Drop 条件：点属于满足条件的体素，且随机数命中 Drop 概率
#         drop_condition = torch.logical_and(is_in_selected_voxel, random_tensor < inner_drop_prob)

#         mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#         mask[drop_condition] = 0.0

#         # --- [5. 补偿与应用] ---
#         actual_keep_ratio = mask.mean()
#         if actual_keep_ratio < 1.0:
#             rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#             opacity = opacity * (mask * rescale_factor)[:, None]
#     if is_train and iteration is not None:
#             # --- [1. 参数配置] ---
#             #import random
#             #ratio=random.uniform(0.001,0.003)
#             progress = iteration / 10000.0


#             #voxel_ratio = 0.001 #*(1-progress)+ 0.008 * progress
#             #voxel_ratio =0.001#*min(2*(iteration/10000),1)
#             curr_xyz = means3D.detach()
#             num_points = curr_xyz.shape[0]
#             min_points_per_voxel =20*(iteration/10000)   # 新增：定义体素点数阈值（≥20）

#             # --- [2. 空间体素化与索引获取] ---
#             # 计算场景的包围盒
#             q_max = torch.quantile(curr_xyz, 0.98, dim=0)
#             q_min = torch.quantile(curr_xyz, 0.02, dim=0)
#             # 计算场景的物理长宽高
#             scene_dims = (q_max - q_min).clamp(min=0.001) # 防止除零
#             scene_extent = scene_dims.max()
            
#             # =========================================================
#             # [核心修改] 动态计算 voxel_size
#             # =========================================================
#             # 设定目标：我们希望每个非空体素里平均大概有 target_cnt 个点
#             target_cnt = 5 # 设为30以确保大多数体素能满足 >=20 的阈值
            
#             # 1. 计算当前场景的“总主要体积”
#             total_volume = scene_dims.prod()
            
#             # 2. 计算平均全局密度 (点数 / 体积)
#             # 注意：3DGS的点通常分布在表面(2D)，直接用体积除会低估密度，
#             # 所以这里计算出的 size 是一个“理论下界”。
#             global_density = num_points / (total_volume + 1e-6)
            
#             # 3. 反推理论体素体积： Vol = target_cnt / density
#             theoretical_voxel_vol = target_cnt / (global_density + 1e-6)
#             theoretical_voxel_size = theoretical_voxel_vol ** (1/3)
            
#             # 4. [关键] 表面稀疏度修正系数 (Surface Correction Factor)
#             # 因为点云是空心的（只在物体表面），实际需要的体素比纯体积计算的要大
#             # 经验值：2.0 ~ 4.0 之间。如果发现 dense 体素还是太少，调大这个值。
#             #sparsity_factor = 2.0 
            
#             dynamic_voxel_size = theoretical_voxel_size #* sparsity_factor
            
#             # 限制一下最小/最大尺寸，防止极端情况爆炸
#             min_size = scene_extent * 0.001
#             max_size = scene_extent * 0.1
#             dynamic_voxel_size = torch.clamp(dynamic_voxel_size, min_size, max_size)
#             # =========================================================
#             voxel_coords = (curr_xyz / dynamic_voxel_size).int()

#             # unique_voxels: 唯一体素的坐标
#             # inverse_indices: 每个点所属体素在 unique_voxels 中的索引
#             unique_voxels, inverse_indices = torch.unique(
#                 voxel_coords, dim=0, return_inverse=True
#             )

#             num_unique_voxels = unique_voxels.shape[0]

#             # --- [3. 筛选满足点数阈值（≥20）且随机挑选 X% 的体素] ---
#             # 步骤3.1：统计每个唯一体素包含的点数
#             counts = torch.bincount(inverse_indices, minlength=num_unique_voxels)  # 确保长度与唯一体素一致

#             # 步骤3.2：新增：生成「体素点数≥20」的掩码（布尔张量，长度=num_unique_voxels）
#             voxel_has_enough_points = counts>=target_cnt#>= min_points_per_voxel

#             # 步骤3.3：设定随机挑选体素的比例
#             voxel_selection_ratio = 0.25*(iteration/10000)
#             # 优化：动态获取设备，避免硬编码cuda（适配CPU/GPU）
#             voxel_random_vals = torch.rand(num_unique_voxels, device=curr_xyz.device)

#             # 步骤3.4：生成「随机选中体素」的掩码
#             voxel_randomly_selected = voxel_random_vals < voxel_selection_ratio

#             # 步骤3.5：组合两个掩码：仅保留「点数≥20 且 被随机选中」的体素（核心修改）
#             selected_voxels_mask =torch.logical_and(voxel_has_enough_points, voxel_randomly_selected)
# #             selected_indices = None
# #             if iteration <= 6000: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #                 selected_indices = pc.restrict_sh_degree_by_points(iteration, random_pct=0.25*(iteration/10000))

# #             ======================================
# #             新增：体素统计打印核心代码（按需隔代打印，格式完全匹配要求）
# #             ======================================
# #             if iteration % 100 == 0:  # 每100次迭代打印一次，避免刷屏，可改200/500
# #                 # 1. 基础统计：总体素、高密度体素、待丢弃体素（选中体素）
# #                 total_vox = num_unique_voxels
# #                 dense_vox = voxel_has_enough_points.sum().item()
# #                 drop_vox = selected_voxels_mask.sum().item()
# #                 # 2. 平均点数统计：所有体素、高密度体素（鲁棒性处理，避免空张量）
# #                 avg_all = counts.float().mean().item()
# #                 avg_dense = counts[voxel_has_enough_points].float().mean().item() if voxel_has_enough_points.any() else 0.0
# #                 # 3. 格式化打印：和你要求的格式完全一致，对齐更美观
# #                 print(f"[Iter {iteration:>4d}] 体素统计：总 {total_vox:>4d} | 密 {dense_vox:>4d} | 丢 {drop_vox:>4d} | 所有体素平均点数: {avg_all:>5.2f} | 高密度体素平均点数: {avg_dense:>5.2f}")

#             # 步骤3.6：仅计算满足条件的体素的点数均值（优化打印）
#             if selected_voxels_mask.any():
#                 selected_counts = counts[selected_voxels_mask]
#                 avg_points_in_selected = selected_counts.float().mean().item()
#                 #print(f"Average points in SELECTED voxels (≥20 points): {avg_points_in_selected:.2f}")
#             #else:
#                 #print("No voxels meet the condition (≥20 points and randomly selected)")

#             # --- [4. 执行 Drop 操作] ---
#             # 将「体素级掩码」映射回「点级掩码」：哪些点属于满足条件的体素
#             is_in_selected_voxel = selected_voxels_mask[inverse_indices]

#             # 定义在被选中的体素内部的 Drop 概率（1.0 表示删掉体素内所有点）
#             inner_drop_prob = 1
#             random_tensor = torch.rand(num_points, device=curr_xyz.device)

#             # 最终 Drop 条件：点属于满足条件的体素，且随机数命中 Drop 概率
#             drop_condition = torch.logical_and(is_in_selected_voxel, random_tensor < inner_drop_prob)
            
#             # 可选：验证体素内点是否全量被标记（取第一个高密度体素，查看内部点的标记情况）
#             if voxel_has_enough_points.any():
#                 # 取第一个高密度体素的索引
#                 first_dense_voxel_idx = torch.nonzero(voxel_has_enough_points)[0].item()
#                 # 取该体素内所有点的索引
#                 points_in_first_dense_voxel = (inverse_indices == first_dense_voxel_idx)
#                 # 查看这些点是否全部被标记为待Drop
#                 all_dropped = drop_condition[points_in_first_dense_voxel].all().item()
#                 print(f"[验证] 第一个高密度体素内点是否全部被Drop: {'是' if all_dropped else '否'} | 该体素内点数: {points_in_first_dense_voxel.sum().item()}")

#             mask = torch.ones(num_points, dtype=torch.float32, device=curr_xyz.device)
#             mask[drop_condition] = 0.0

#             # --- [5. 补偿与应用] ---
#             actual_keep_ratio = mask.mean()
#             if actual_keep_ratio < 1.0:
#                 rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#                 opacity = opacity * (mask * rescale_factor)[:, None]
#     if is_train and iteration is not None:
#         # --- [1. 超参数配置：集中管理，3便调参] ---
#         progress = iteration / 10000.0  # 训练进度(0~1)
#         num_points = means3D.shape[0]#.detach().shape[0]
#         # 体素筛选核心参数
#         target_points_per_voxel = 20*progress    # 目标体素平均点数（动态体素尺寸的核心依据）
#         dense_voxel_threshold =  20*progress     # 高密度体素阈值（与target_cnt保持一致，命名更清晰）
#         voxel_rand_ratio_coeff = 0.25  # 随机选中比例系数（0~1，控制随机稀疏幅度）
#         # 3DGS表面点云修正：因点云分布在表面(2D)，需放大体素尺寸（经验值2.0~4.0，关闭设为1.0）
#         surface_sparsity_factor = 1.0#*progress
#         # 体素尺寸限制：防止极端场景下体素过大/过小
#         voxel_size_min_ratio = 0.001
#         voxel_size_max_ratio = 0.1

#         # --- [2. 动态体素尺寸计算：适配3DGS表面点云分布] ---
#         curr_xyz = means3D#.detach()
#         # 计算场景有效包围盒（分位数过滤离群点，更鲁棒）
#         q_max = torch.quantile(curr_xyz, 0.98, dim=0, keepdim=True)  # 保持维度，避免广播问题
#         q_min = torch.quantile(curr_xyz, 0.02, dim=0, keepdim=True)
#         scene_dims = (q_max - q_min).clamp(min=1e-6)  # 除零保护，替代0.001更严谨
#         scene_extent = scene_dims.max()  # 场景最大延伸尺寸

#         # 2.1 计算全局密度（点数/场景体积，3DGS表面点云会低估密度，仅作理论依据）
#         total_volume = scene_dims.prod()
#         global_density = num_points / (total_volume + 1e-6)  # 除零保护

#         # 2.2 反推理论体素尺寸：基于目标平均点数计算
#         theoretical_voxel_vol = target_points_per_voxel / (global_density + 1e-6)
#         theoretical_voxel_size = theoretical_voxel_vol ** (1/3)

#         # 2.3 3DGS表面稀疏度修正 + 尺寸限制（核心优化，适配表面点云）
#         dynamic_voxel_size = theoretical_voxel_size * surface_sparsity_factor
#         voxel_size_min = scene_extent * voxel_size_min_ratio
#         voxel_size_max = scene_extent * voxel_size_max_ratio
#         dynamic_voxel_size = torch.clamp(dynamic_voxel_size, voxel_size_min, voxel_size_max)

#         # --- [3. 空间体素化：平移到非负区间，避免负坐标体素划分失真（核心修复）] ---
#         curr_xyz_shifted = curr_xyz - q_min  # 平移到以场景最小点为原点，所有坐标≥0
#         voxel_coords = (curr_xyz_shifted / dynamic_voxel_size).int()  # 体素坐标非负，划分更合理

#         # 获取唯一体素坐标 + 原始点到唯一体素的索引映射
#         unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
#         num_unique_voxels = unique_voxels.shape[0]
#         device = curr_xyz.device  # 动态获取设备，适配CPU/GPU/多GPU，避免硬编码

#         # --- [4. 高密度体素筛选 + 随机稀疏掩码生成] ---
#         # 4.1 统计每个唯一体素内的点数（标准高效写法，无问题）
#         counts = torch.bincount(inverse_indices, minlength=num_unique_voxels)
#         # 4.2 生成高密度体素掩码（点数≥阈值，命名清晰，无注释冗余）
#         voxel_is_dense = counts >= dense_voxel_threshold#[体素num,]
#         # 4.3 动态计算随机选中比例（随训练进度从0提升到voxel_rand_ratio_coeff）
#         voxel_rand_select_ratio = voxel_rand_ratio_coeff * progress
#         # 4.4 生成随机掩码（动态设备，无硬编码）
#         voxel_rand_vals = torch.rand(num_unique_voxels, device=device)
#         voxel_is_rand_selected = voxel_rand_vals < voxel_rand_select_ratio
#         # 4.5 组合掩码：仅丢弃「高密度 + 随机选中」的体素（保留and逻辑，贴合稀疏需求）
#         voxels_to_drop_mask = torch.logical_and(voxel_is_dense, voxel_is_rand_selected)#[体素num,]

#         # --- [5. 全量体素统计打印：隔代输出，关键指标全覆盖（方便调参）] ---
#         if iteration % 100 == 0:  # 每100次迭代打印，可根据训练速度调整
#             # 基础体素统计
#             total_vox = num_unique_voxels
#             dense_vox = voxel_is_dense.sum().item()
#             drop_vox = voxels_to_drop_mask.sum().item()
#             # 平均点数统计（鲁棒性处理：空张量时赋值0.0）
#             avg_points_all = counts.float().mean().item() if num_unique_voxels > 0 else 0.0
#             avg_points_dense = counts[voxel_is_dense].float().mean().item() if voxel_is_dense.any() else 0.0
#             # 待丢弃点统计（映射到点级，直观看到Drop幅度）
#             points_to_drop = voxels_to_drop_mask[inverse_indices].sum().item()
#             drop_points_ratio = points_to_drop / num_points if num_points > 0 else 0.0
#             # 格式化打印：对齐美观，指标全覆盖（体素尺寸+体素统计+点统计）
#             #print(f"[Iter {iteration:>4d}] 体素尺寸: {dynamic_voxel_size.item():.6f} | 体素统计：总 {total_vox:>4d} | 密 {dense_vox:>4d} | 丢 {drop_vox:>4d} | 所有体素平均点数: {avg_points_all:>5.2f} | 高密度体素平均点数: {avg_points_dense:>5.2f} | 待丢点数: {points_to_drop:>5d} ({drop_points_ratio:.2%})")

#         # --- [6. 执行Drop操作：移除冗余逻辑，实现体素内点100%全量Drop（核心优化）] ---
#         # 体素级掩码映射到点级：哪些点属于待丢弃体素
#         points_in_drop_voxel = voxels_to_drop_mask[inverse_indices]#
#         # 直接用点级掩码作为Drop条件，移除冗余的随机张量+and判断，确保全量Drop
#         drop_condition = points_in_drop_voxel

#         # --- [7. 真实验证：修复torch.nonzero坑，确保高密度体素点数≥阈值+全量Drop] ---
#         # if voxel_is_dense.any():
#         #     # 安全获取高密度体素索引：用torch.where替代torch.nonzero，避免1维张量取值坑
#         #     dense_voxel_indices = torch.where(voxel_is_dense)[0]
#         #     first_dense_vox_idx = dense_voxel_indices[0].item()
#         #     # 双重校验：确保取到的体素是真正的高密度体素（理论上不会触发断言）
#         #     assert counts[first_dense_vox_idx] >= dense_voxel_threshold, \
#         #         f"校验失败：体素{first_dense_vox_idx}点数{counts[first_dense_vox_idx].item()} < 阈值{dense_voxel_threshold}"
#         #     # 取该体素内所有点的掩码
#         #     points_in_first_dense = inverse_indices == first_dense_vox_idx
#         #     # 验证该体素内点是否100%被标记为Drop
#         #     is_all_dropped = drop_condition[points_in_first_dense].all().item()
#         #     dense_vox_actual_points = points_in_first_dense.sum().item()
#         #     # 打印验证结果
#         #     print(f"[验证] 第1个高密度体素(索引{first_dense_vox_idx})：点数={dense_vox_actual_points} | 内部点是否全量Drop: {'是' if is_all_dropped else '否'}")

#         # --- [8. 生成保留掩码 + 不透明度补偿：保证点云整体能量不变] ---
#         mask = torch.ones(num_points, dtype=torch.float32, device=device)
#         mask[drop_condition] = 0.0  # 待丢弃点掩码置0
#         # 计算实际保留比例，做能量补偿（避免Drop后点云整体透明度变化）
#         actual_keep_ratio = mask.mean()
#         if actual_keep_ratio < 1.0 - 1e-6:  # 浮点误差容忍，避免无Drop时的无效计算
#             rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#             opacity = opacity * (mask * rescale_factor)[:, None]
#     if is_train and iteration is not None:
#             with torch.no_grad(): # 确保计算过程不记录梯度，节省显存
#                 # --- [1. 参数准备] ---
#                 # 假设你能够获取当前高斯的 scaling，形状为 [N, 3]
#                 # 如果代码上下文中没有 scales，请取消下面这行的注释并根据你的对象名称调整
#                 #scales = gaussians.get_scaling() 

#                 curr_xyz = means3D.detach()
#                 num_points = curr_xyz.shape[0]

#                 # 动态调整体素大小
#                 voxel_base_ratio = 0.003
#                 voxel_ratio = voxel_base_ratio * min(2 * (iteration / 10000), 1)

#                 scene_extent = (curr_xyz.max(dim=0)[0] - curr_xyz.min(dim=0)[0]).max()
#                 dynamic_voxel_size = scene_extent * voxel_ratio

#                 # --- [2. 空间体素化 (含 Grid Jittering 优化)] ---
#                 # 优化：引入随机偏移 (Jitter)，打破固定网格的刚性边界
#                 # 这能让 Drop 操作在多次迭代中覆盖连续空间，而不是死板的方块
#                 voxel_shift = torch.rand(3, device="cuda") * dynamic_voxel_size
#                 voxel_coords = ((curr_xyz + voxel_shift) / dynamic_voxel_size).int()

#                 # 获取唯一体素索引
#                 unique_voxels, inverse_indices = torch.unique(
#                     voxel_coords, dim=0, return_inverse=True
#                 )
#                 num_unique_voxels = unique_voxels.shape[0]

#                 # --- [3. 计算体素属性：点数与平均体积] ---
#                 # 3.1 统计每个体素的点数
#                 voxel_counts = torch.bincount(inverse_indices, minlength=num_unique_voxels).float()

#                 # 3.2 计算每个点的近似体积 (长*宽*高)
#                 point_volumes = torch.prod(scales, dim=1) # [N]

#                 # 3.3 聚合计算每个体素的总体积
#                 voxel_total_vol = torch.zeros(num_unique_voxels, device="cuda")
#                 # scatter_add_: 将 point_volumes 加到对应的 inverse_indices 位置
#                 voxel_total_vol.scatter_add_(0, inverse_indices, point_volumes)

#                 # 3.4 计算体素内的平均点体积 (Average Volume per Point in Voxel)
#                 voxel_avg_vol = voxel_total_vol / (voxel_counts + 1e-6)

#                 # --- [4. 自适应筛选策略] ---
#                 # 策略A: 密度阈值 (体素内点数必须足够多，才考虑 Drop)
#                 min_points_per_voxel = 25
#                 mask_enough_points = voxel_counts >= min_points_per_voxel

#                 # 策略B: 随机时间掩码 (控制整体 Drop 的频率)
#                 selection_ratio = 0.1 * (iteration / 10000)
#                 rand_vals = torch.rand(num_unique_voxels, device="cuda")
#                 mask_selected_time = rand_vals < selection_ratio

#                 # 组合体素级掩码
#                 target_voxel_mask = torch.logical_and(mask_enough_points, mask_selected_time)

#                 # --- [5. 执行细粒度 Drop] ---
#                 # 将体素属性映射回点级
#                 point_in_target_voxel = target_voxel_mask[inverse_indices]

#                 if point_in_target_voxel.any():
#                     # 获取每个点所属体素的平均体积
#                     point_local_avg_vol = voxel_avg_vol[inverse_indices]

#                     # --- [核心逻辑：基于尺度的动态概率] ---
#                     # 归一化体积到 [0, 1]，用于计算概率
#                     # 使用 quantile 避免极值影响，增强鲁棒性
#                     vol_min = point_local_avg_vol.min()
#                     vol_max = torch.quantile(point_local_avg_vol, 0.95) # 取95分位数为最大值
#                     norm_vol = (point_local_avg_vol - vol_min) / (vol_max - vol_min + 1e-7)
#                     norm_vol = torch.clamp(norm_vol, 0, 1)

#                     # 动态概率公式：
#                     # 体积越小(细节) -> drop_prob 越接近 0.1 (保护)
#                     # 体积越大(背景) -> drop_prob 越接近 0.6 (激进)
#                     base_prob = 0.1
#                     variable_prob = 1.0
#                     adaptive_probs = base_prob + variable_prob * norm_vol

#                     # 生成随机数进行 Drop
#                     random_tensor = torch.rand(num_points, device="cuda")

#                     # Drop 条件：
#                     # 1. 点在被选中的体素中
#                     # 2. 随机数小于该点的自适应概率
#                     should_drop = torch.logical_and(point_in_target_voxel, random_tensor < adaptive_probs)

#                     # 生成最终 Mask (0为丢弃，1为保留)
#                     mask = torch.ones(num_points, dtype=torch.float32, device="cuda")
#                     mask[should_drop] = 0.0

#                     # --- [6. 能量补偿 (Rescale)] ---
#                     # 计算实际保留率
#                     actual_keep_ratio = mask.mean()

#                     if actual_keep_ratio < 1.0:
#                         # 简单补偿：整体增强，保持总不透明度期望一致
#                         rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#                         opacity = opacity * (mask * rescale_factor)[:, None]

#                         # 调试打印 (可选)
#                         # if iteration % 1000 == 0:
#                         #     print(f"[Iter {iteration}] Drop Ratio: {(1-actual_keep_ratio):.4f}, "
#                         #           f"Voxel Avg Vol Range: {vol_min:.2e}~{vol_max:.2e}")
#                 else:
#                     # 如果没有体素被选中，不做任何操作
#                     pass
#     if is_train and iteration is not None:
#         # --- [1. 参数配置] ---
#         # 动态调整体素大小
#         voxel_ratio = 0.003 #* min(2 * (iteration / 10000), 1)

#         curr_xyz = means3D.detach()
#         num_points = curr_xyz.shape[0]

#         # 设定阈值：你提到的“点数为20”通常指“至少20”以消除高密度冗余
#         min_points_per_voxel = 20 

#         # --- [2. 空间体素化 (保持高精度)] ---
#         # 使用分位数计算边界，防止离群点导致 bounding box 过大，提高体素化精度
#         q_max = torch.quantile(curr_xyz, 0.98, dim=0)
#         q_min = torch.quantile(curr_xyz, 0.02, dim=0)
#         scene_extent = (q_max - q_min).max()

#         dynamic_voxel_size = scene_extent * voxel_ratio

#         # 获取体素坐标
#         voxel_coords = (curr_xyz / dynamic_voxel_size).int()

#         # unique_voxels: 唯一体素坐标
#         # inverse_indices: 每个点对应第几个 unique_voxel (核心映射)
#         unique_voxels, inverse_indices = torch.unique(
#             voxel_coords, dim=0, return_inverse=True
#         )
#         num_unique_voxels = unique_voxels.shape[0]

#         # --- [3. 核心逻辑：精准筛选需 Drop 的体素] ---

#         # 3.1 统计每个体素内的点数
#         # counts 维度 = [num_unique_voxels]
#         counts = torch.bincount(inverse_indices, minlength=num_unique_voxels)

#         # 3.2 条件A：体素密度阈值 (点数 >= 20)
#         # 逻辑：只有点数足够密集的体素才会被考虑删除，保护稀疏细节
#         mask_density_high = counts >= min_points_per_voxel

#         # 3.3 条件B：基于迭代次数的随机选择概率
#         # 逻辑：iteration 越大，删除力度越大，上限设为 0.25 (即 25%)
#         current_drop_ratio = 0.25 * (iteration / 10000)

#         # 生成每个体素的随机值 (0.0 ~ 1.0)
#         voxel_random_vals = torch.rand(num_unique_voxels, device="cuda")
#         mask_random_hit = voxel_random_vals < current_drop_ratio

#         # 3.4 组合条件：最终决定哪些 体素 需要被移除
#         # 只有同时满足 "密度够大" AND "运气不好被抽中" 的体素才会被标记
#         voxels_to_drop_mask = torch.logical_and(mask_density_high, mask_random_hit)

#         # --- [4. 映射回点并执行 Drop] ---

#         # 关键步骤：利用 inverse_indices 将“体素掩码”广播回“所有点”
#         # points_to_drop 维度 = [num_points]
#         points_to_drop = voxels_to_drop_mask[inverse_indices]

#         # 生成最终保留掩码 (取反，True 表示保留)
#         mask = ~points_to_drop

#         # --- [5. 补偿与应用] ---
#         # 转换为 float 以便计算和乘法
#         mask = mask.float()
#         actual_keep_ratio = mask.mean()

#         if actual_keep_ratio < 1.0:
#             # 能量补偿：剩下的点稍微变实一点，弥补删掉的体积感
#             rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#             opacity = opacity * (mask * rescale_factor)[:, None]
#     if is_train and iteration is not None:
#         # --- [1. 参数配置] ---
#         # 动态计算体素大小 (保持原有逻辑)
#         voxel_ratio = 0.003 #* min(2 * (iteration / 10000), 1)

#         curr_xyz = means3D.detach()
#         num_points = curr_xyz.shape[0]

#         # 定义阈值
#         min_points_per_voxel = 15  # 体素内点数阈值

#         # 定义随机选择点的比例
#         point_selection_ratio = 0.25 * (iteration / 10000)

#         # --- [2. 空间体素化] ---
#         # 计算场景范围
#         q_max = torch.quantile(curr_xyz, 0.98, dim=0)
#         q_min = torch.quantile(curr_xyz, 0.02, dim=0)
#         scene_extent = (q_max - q_min).max()

#         dynamic_voxel_size = scene_extent * voxel_ratio

#         # 计算每个点所属的体素坐标
#         voxel_coords = (curr_xyz / dynamic_voxel_size).int()

#         # inverse_indices: 长度为 [num_points]，表示每个点属于第几个 unique_voxel
#         unique_voxels, inverse_indices = torch.unique(
#             voxel_coords, dim=0, return_inverse=True
#         )
#         num_unique_voxels = unique_voxels.shape[0]

#         # --- [3. 逻辑核心：点选触发体素删除] ---

#         # 步骤 3.1: 统计体素密度
#         # counts: [num_unique_voxels]
#         counts = torch.bincount(inverse_indices, minlength=num_unique_voxels)

#         # 标记哪些体素是“拥挤”的 (点数 > 20)
#         # voxel_is_dense: [num_unique_voxels] (Boolean)
#         voxel_is_dense = counts > min_points_per_voxel

#         # 步骤 3.2: 随机选择点 (作为“触发器”)
#         # 生成一个随机掩码，选中 random% 的点
#         # point_random_mask: [num_points] (Boolean)
#         random_vals = torch.rand(num_points, device="cuda")
#         point_random_mask = random_vals < point_selection_ratio

#         # 步骤 3.3: 确定哪些体素被“击中”了
#         # 我们需要知道：哪些体素内部包含至少一个被选中的点？

#         # 方法：获取被选中点的体素索引
#         hit_voxel_indices = inverse_indices[point_random_mask]

#         # 创建一个全False的体素掩码，将被击中的体素置为True
#         # voxel_is_hit: [num_unique_voxels] (Boolean)
#         voxel_is_hit = torch.zeros(num_unique_voxels, dtype=torch.bool, device="cuda")
#         # 使用 index_fill_ 将被击中的体素索引设为 True
#         voxel_is_hit.index_fill_(0, hit_voxel_indices, True)

#         # 步骤 3.4: 组合条件 (交集)
#         # 只有同时满足：(1) 体素被随机点击中 AND (2) 体素本身点数 > 20
#         # voxels_to_drop: [num_unique_voxels]
#         voxels_to_drop = torch.logical_or(voxel_is_dense, voxel_is_hit)

#         # --- [4. 执行 Drop] ---

#         # 将体素级的决策映射回所有点
#         # 如果一个点属于 voxels_to_drop 中的体素，它就被删除
#         points_to_drop_mask = voxels_to_drop[inverse_indices]

#         # 生成最终保留掩码 (~ 取反，因为 1 是保留，0 是删除)
#         mask = ~points_to_drop_mask
#         mask = mask.float()

#         # --- [5. 补偿与应用] ---
#         actual_keep_ratio = mask.mean()

#         if actual_keep_ratio < 1.0:
#             # 视觉补偿：被删掉的区域，剩下的点（如果有的话，但这里是整块删，所以是对整体场景的补偿）
#             # 注意：这里是整块删，周围会有空洞，补偿主要是为了防止loss突变
#             rescale_factor = 1.0 / (actual_keep_ratio + 1e-7)
#             opacity = opacity * (mask * rescale_factor)[:, None]
        
        # 调试打印 (可选)
        # if iteration % 1000 == 0:
        #     dropped_voxels = voxels_to_drop_mask.sum().item()
        #     print(f"Iter {iteration}: Dropped {dropped_voxels} dense voxels.")
#     if is_train and iteration is not None : #and iteration >= 500:  # 跳过初期
#         num_points = means3D.shape[0]
#         device = means3D.device
#         progress = min(iteration / 10000.0, 1.0)

#         # === 场景范围 ===
#         q_max = torch.quantile(means3D, 0.98, dim=0)
#         q_min = torch.quantile(means3D, 0.02, dim=0)
#         scene_dims = (q_max - q_min).clamp(min=1e-6)
#         scene_extent = scene_dims.max()
#         scene_volume = scene_dims.prod()

#         # === 自适应体素大小（核心改进）===
#         target_points_per_voxel = 22  # 固定目标点数,参数1
#         current_density = num_points / (scene_volume + 1e-6)
#         adaptive_voxel_size = (target_points_per_voxel / (current_density + 1e-6)) ** (1/3)

#         # 表面点云修正 + 范围限制
#         surface_factor = 1.0  # 3DGS 表面分布修正
#         voxel_size = adaptive_voxel_size * surface_factor
#         voxel_size = torch.clamp(voxel_size, 
#                                  scene_extent * 0.01, 
#                                  scene_extent * 0.1)

#         # === 自适应 drop 参数 ===
#         # 增密期 (iteration < 7000) 降低强度，参数2
#         if iteration < 7000:
#             adj_progress = progress * 0.6 #前期选择点数较少
#         else:
#             adj_progress = progress

#         voxel_select_ratio = 0.25 * adj_progress#参数3
#         inner_drop_rate = 0.4 + 0.4 * adj_progress#参数4

#         # === 体素化 ===
#         shifted_xyz = means3D - q_min
#         voxel_coords = (shifted_xyz / voxel_size).int()
#         unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
#         num_voxels = unique_voxels.shape[0]

#         # === 密度筛选（百分位法，自适应）===
#         counts = torch.bincount(inverse_indices, minlength=num_voxels)
#         # 取 top 30% 作为高密度
#         dense_threshold = torch.quantile(counts.float(), 0.8).item()#参数5
#         dense_threshold = max(dense_threshold, 10)  # 最低阈值保护，参数6

#         is_dense = counts >= dense_threshold
#         is_selected = torch.rand(num_voxels, device=device) < voxel_select_ratio
#         target_voxels = is_dense & is_selected

#         # === 映射到点 + 部分 drop ===
#         points_in_target = target_voxels[inverse_indices]
#         inner_rand = torch.rand(num_points, device=device)
#         drop_mask = points_in_target & (inner_rand < inner_drop_rate)

#         # === Compensation ===
#         compensation = torch.ones(num_points, device=device)
#         compensation[drop_mask] = 0.0

#         keep_ratio = compensation.mean()
#         if keep_ratio < 1.0 - 1e-6:
#             compensation = compensation / (keep_ratio + 1e-7)

#         # 应用到 opacity
#         opacity = opacity * compensation.unsqueeze(-1)
#2-2 SH
    if is_train and iteration is not None: #and iteration >= 500:
        num_points = means3D.shape[0]
        device = means3D.device
        progress = min(iteration / 10000.0, 1.0)

        # === 场景范围 ===
        q_max = torch.quantile(means3D, 0.98, dim=0)
        q_min = torch.quantile(means3D, 0.02, dim=0)
        scene_dims = (q_max - q_min).clamp(min=1e-6)
        scene_extent = scene_dims.max()
        scene_volume = scene_dims.prod()

        # === 自适应体素大小 ===
        target_points_per_voxel = 20
        current_density = num_points / (scene_volume + 1e-6)
        adaptive_voxel_size = (target_points_per_voxel / (current_density + 1e-6)) ** (1/3)

        surface_factor = 1.0
        voxel_size = adaptive_voxel_size * surface_factor
        voxel_size = torch.clamp(voxel_size, 
                                 scene_extent * 0.01, 
                                 scene_extent * 0.1)

        # === 自适应 drop 参数 ===
        if iteration < 7000:
            adj_progress = progress * 0.6
        else:
            adj_progress = progress

        voxel_select_ratio = 0.25 * adj_progress
        inner_drop_rate = 0.4 + 0.4 * adj_progress

        # === 体素化 ===
        shifted_xyz = means3D - q_min
        voxel_coords = (shifted_xyz / voxel_size).int()
        unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        num_voxels = unique_voxels.shape[0]

        # === 密度筛选 ===
        counts = torch.bincount(inverse_indices, minlength=num_voxels)
        dense_threshold = torch.quantile(counts.float(), 0.8).item()
        dense_threshold = max(dense_threshold, 10)

        is_dense = counts >= dense_threshold
        is_selected = torch.rand(num_voxels, device=device) < voxel_select_ratio
        target_voxels = is_dense & is_selected

        # === 映射到点 + drop mask ===
        points_in_target = target_voxels[inverse_indices]
        inner_rand = torch.rand(num_points, device=device)
        drop_mask = points_in_target & (inner_rand < inner_drop_rate)

        # === 【关键】调用SH限制函数，传入drop的点索引 ===
        if iteration < 6000:
            drop_indices = torch.where(drop_mask)[0]
            if len(drop_indices) > 0:
                # 对被drop的点同时限制SH阶数
                pc.restrict_sh_degree_by_points(
                    iteration, 
                    selected_indices=drop_indices,
                    random_pct=0.0  # 不使用随机，直接用传入的索引
                )
            else: print("error!!!")

        # === Compensation ===
        compensation = torch.ones(num_points, device=device)
        compensation[drop_mask] = 0.0

        keep_ratio = compensation.mean()
        if keep_ratio < 1.0 - 1e-6:
            compensation = compensation / (keep_ratio + 1e-7)

        opacity = opacity * compensation.unsqueeze(-1)


        # === Debug 日志 ===
        # if iteration % 1000 == 0:
        #     print(f"[VoxelDrop] iter={iteration}, pts={num_points}, "
        #           f"voxel_size={voxel_size.item():.4f}, "
        #           f"threshold={dense_threshold:.1f}, "
        #           f"dropped={drop_mask.sum().item()} ({drop_mask.float().mean()*100:.1f}%)")
    # 调用光栅化器渲染图像
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # 归一化并返回结果
    rendered_image = rendered_image.clamp(0, 1)
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii
    }
