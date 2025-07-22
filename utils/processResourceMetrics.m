function [resource_util, alloc_balance] = processResourceMetrics(info, config, expected_cols)
%PROCESSRESOURCEMETRICS 处理资源利用率和分配均衡性指标
% 
% 输入参数:
%   info - 环境信息结构体，包含资源分配和使用情况
%   config - 配置参数结构体
%   expected_cols - 期望的列数（防御者数量）
%
% 输出参数:
%   resource_util - 资源利用率向量 [1 x expected_cols]
%   alloc_balance - 分配均衡性向量 [1 x expected_cols]

try
    % 初始化输出变量
    resource_util = zeros(1, expected_cols);
    alloc_balance = zeros(1, expected_cols);
    
    % 检查 info 结构体是否包含必要字段
    if ~isstruct(info)
        warning('processResourceMetrics:InvalidInput', 'info 参数不是有效的结构体');
        return;
    end
    
    % 获取资源分配信息
    if isfield(info, 'resource_allocation') && ~isempty(info.resource_allocation)
        current_allocation = info.resource_allocation(:)'; % 转换为行向量
    elseif isfield(info, 'defender_deployment') && ~isempty(info.defender_deployment)
        current_allocation = info.defender_deployment(:)';
    else
        % 如果没有资源分配信息，使用均匀分配作为默认值
        if isfield(config, 'system') && isfield(config.system, 'n_stations')
            n_stations = config.system.n_stations;
        else
            n_stations = 5; % 默认5个站点
        end
        current_allocation = ones(1, n_stations) / n_stations;
    end
    
    % 获取总资源数
    if isfield(config, 'system') && isfield(config.system, 'total_resources')
        total_resources = config.system.total_resources;
    else
        total_resources = 100; % 默认总资源
    end
    
    % 计算资源利用率
    % 资源利用率 = 实际使用的资源 / 可用总资源
    total_used = sum(current_allocation);
    if total_resources > 0
        base_utilization = min(1.0, total_used / total_resources);
    else
        base_utilization = 0.0;
    end
    
    % 根据防御者数量复制资源利用率
    for i = 1:expected_cols
        resource_util(i) = base_utilization;
    end
    
    % 计算分配均衡性
    % 使用基尼系数的简化版本来衡量分配均衡性
    % 均衡性越高，值越接近1；不均衡时值接近0
    if length(current_allocation) > 1 && sum(current_allocation) > 0
        % 标准化分配比例
        normalized_allocation = current_allocation / sum(current_allocation);
        
        % 计算均匀分配的基准
        uniform_allocation = ones(size(normalized_allocation)) / length(normalized_allocation);
        
        % 计算与均匀分配的偏差
        deviations = abs(normalized_allocation - uniform_allocation);
        max_possible_deviation = 1 - 1/length(normalized_allocation);
        
        if max_possible_deviation > 0
            balance_score = 1 - sum(deviations) / (2 * max_possible_deviation);
        else
            balance_score = 1.0;
        end
        
        % 确保分配均衡性在[0,1]范围内
        balance_score = max(0, min(1, balance_score));
    else
        balance_score = 0.0; % 如果没有有效分配，均衡性为0
    end
    
    % 根据防御者数量复制分配均衡性
    for i = 1:expected_cols
        alloc_balance(i) = balance_score;
    end
    
    % 添加一些变化以区分不同的防御者（可选）
    if expected_cols > 1
        % 为不同的防御者添加小的随机变化（±5%）
        variation_factor = 0.05;
        for i = 1:expected_cols
            resource_util(i) = resource_util(i) * (1 + variation_factor * (rand - 0.5));
            alloc_balance(i) = alloc_balance(i) * (1 + variation_factor * (rand - 0.5));
        end
        
        % 确保值在合理范围内
        resource_util = max(0, min(1, resource_util));
        alloc_balance = max(0, min(1, alloc_balance));
    end
    
    % 处理特殊情况
    resource_util(isnan(resource_util)) = 0;
    alloc_balance(isnan(alloc_balance)) = 0;
    
catch ME
    % 错误处理
    warning('processResourceMetrics:Error', '处理资源指标时出错: %s', ME.message);
    
    % 返回默认值
    resource_util = zeros(1, expected_cols);
    alloc_balance = zeros(1, expected_cols);
end

end