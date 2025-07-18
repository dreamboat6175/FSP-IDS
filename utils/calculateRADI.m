function radi = calculateRADI(resource_allocation, optimal_allocation, radi_config)
    %% calculateRADI - 计算RADI(Resource Allocation Deviation Index)指标
    % RADI指标用于评估资源分配相对于最优分配的偏差程度
    % 输入:
    %   resource_allocation - 当前资源分配向量 [1 x n]
    %   optimal_allocation - 最优资源分配向量 [1 x n] (可选)
    %   radi_config - RADI配置结构体 (可选)
    % 输出:
    %   radi - RADI指标值 (标量，越小越好)
    
    try
        % 参数验证和默认值设置
        if nargin < 1 || isempty(resource_allocation)
            error('resource_allocation 不能为空');
        end
        
        % 确保是行向量
        if iscolumn(resource_allocation)
            resource_allocation = resource_allocation';
        end
        
        n_resources = length(resource_allocation);
        
        % 设置默认最优分配（均匀分配）
        if nargin < 2 || isempty(optimal_allocation)
            optimal_allocation = ones(1, n_resources) / n_resources;
        else
            if iscolumn(optimal_allocation)
                optimal_allocation = optimal_allocation';
            end
        end
        
        % 设置默认权重
        if nargin < 3 || isempty(radi_config) || ~isstruct(radi_config)
            % 默认均匀权重
            weights = ones(1, n_resources) / n_resources;
        else
            % 从配置中提取权重
            weights = extractWeightsFromConfig(radi_config, n_resources);
        end
        
        % 确保所有向量长度一致
        min_len = min([length(resource_allocation), length(optimal_allocation), length(weights)]);
        resource_allocation = resource_allocation(1:min_len);
        optimal_allocation = optimal_allocation(1:min_len);
        weights = weights(1:min_len);
        
        % 归一化处理
        [norm_allocation, norm_optimal] = normalizeAllocations(resource_allocation, optimal_allocation);
        
        % 计算RADI指标
        radi = calculateRADICore(norm_allocation, norm_optimal, weights);
        
        % 确保返回标量且在合理范围内
        radi = validateRADIOutput(radi);
        
    catch ME
        warning('calculateRADI 计算出错: %s', ME.message);
        % 返回默认值，避免中断仿真
        radi = 1.0;
    end
end

function weights = extractWeightsFromConfig(radi_config, n_resources)
    %% extractWeightsFromConfig - 从配置中提取权重
    
    try
        % 预定义的权重字段名
        weight_fields = {
            'weight_computation', 'weight_bandwidth', 'weight_sensors', 
            'weight_scanning', 'weight_inspection', 'weight_memory',
            'weight_network', 'weight_storage', 'weight_processing'
        };
        
        weights = [];
        
        % 尝试从配置中提取权重
        for i = 1:min(n_resources, length(weight_fields))
            field = weight_fields{i};
            if isfield(radi_config, field)
                weights(end+1) = radi_config.(field);
            end
        end
        
        % 如果权重不足，用默认值补充
        if length(weights) < n_resources
            default_weight = 1.0 / n_resources;
            weights = [weights, repmat(default_weight, 1, n_resources - length(weights))];
        end
        
        % 如果权重过多，截断
        if length(weights) > n_resources
            weights = weights(1:n_resources);
        end
        
        % 归一化权重
        if sum(weights) > 0
            weights = weights / sum(weights);
        else
            weights = ones(1, n_resources) / n_resources;
        end
        
    catch
        % 如果提取失败，使用均匀权重
        weights = ones(1, n_resources) / n_resources;
    end
end

function [norm_allocation, norm_optimal] = normalizeAllocations(resource_allocation, optimal_allocation)
    %% normalizeAllocations - 归一化资源分配
    
    % 归一化当前分配
    if sum(resource_allocation) > 0
        norm_allocation = resource_allocation / sum(resource_allocation);
    else
        % 如果总和为0，使用均匀分配
        norm_allocation = ones(size(resource_allocation)) / length(resource_allocation);
    end
    
    % 归一化最优分配
    if sum(optimal_allocation) > 0
        norm_optimal = optimal_allocation / sum(optimal_allocation);
    else
        % 如果总和为0，使用均匀分配
        norm_optimal = ones(size(optimal_allocation)) / length(optimal_allocation);
    end
    
    % 确保非负
    norm_allocation = max(0, norm_allocation);
    norm_optimal = max(0, norm_optimal);
end

function radi = calculateRADICore(norm_allocation, norm_optimal, weights)
    %% calculateRADICore - RADI指标的核心计算
    
    % 计算绝对偏差
    deviation = abs(norm_allocation - norm_optimal);
    
    % 加权求和
    radi = sum(weights .* deviation);
    
    % 可选：添加平方惩罚项（对大偏差更敏感）
    % radi = sqrt(sum(weights .* (deviation.^2)));
end

function radi = validateRADIOutput(radi)
    %% validateRADIOutput - 验证和修正RADI输出
    
    % 确保是标量
    if ~isscalar(radi)
        radi = mean(radi(:));
        warning('RADI计算返回非标量值，已转换为均值');
    end
    
    % 处理异常值
    if isnan(radi) || isinf(radi)
        radi = 1.0;  % 默认中等偏差
        warning('RADI计算结果为NaN或Inf，已设为默认值1.0');
    end
    
    % 限制在合理范围内 [0, 2]
    % RADI = 0 表示完美匹配，RADI = 2 表示最大偏差（完全相反的分配）
    radi = max(0, min(radi, 2));
end

%% 辅助函数：创建默认RADI配置
function radi_config = createDefaultRADIConfig(n_resources)
    %% createDefaultRADIConfig - 创建默认的RADI配置
    
    radi_config = struct();
    
    % 默认权重（均匀）
    default_weight = 1.0 / n_resources;
    weight_names = {'weight_computation', 'weight_bandwidth', 'weight_sensors', ...
                   'weight_scanning', 'weight_inspection'};
    
    for i = 1:min(n_resources, length(weight_names))
        radi_config.(weight_names{i}) = default_weight;
    end
    
    % 默认最优分配（均匀分配）
    radi_config.optimal_allocation = ones(1, n_resources) / n_resources;
    
    % 性能阈值
    radi_config.threshold_excellent = 0.1;    % 优秀
    radi_config.threshold_good = 0.3;         % 良好
    radi_config.threshold_acceptable = 0.5;   % 可接受
    
end