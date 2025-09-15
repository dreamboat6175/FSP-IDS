function updatePerformanceMonitor(monitor, iteration, episode_results, config)
    %% updatePerformanceMonitor - 更新性能监控器
    % 输入:
    %   monitor - 性能监控器对象
    %   iteration - 当前迭代编号
    %   episode_results - 当前迭代的episode结果
    %   config - 配置结构体
    
    try
        % 检查monitor是否为空或无效
        if isempty(monitor)
            return;
        end
        
        % 构造监控指标结构
        metrics = struct();
        
        % 基本性能指标
        if isfield(episode_results, 'avg_resource_allocation')
            metrics.resource_allocation = mean(episode_results.avg_resource_allocation, 1);
        else
            metrics.resource_allocation = zeros(1, config.n_stations);
        end
        
        if isfield(episode_results, 'avg_efficiency')
            metrics.resource_efficiency = mean(episode_results.avg_efficiency);
        else
            metrics.resource_efficiency = 0.5; % 默认值
        end
        
        if isfield(episode_results, 'avg_balance')
            metrics.allocation_balance = mean(episode_results.avg_balance);
        else
            metrics.allocation_balance = 0.5; % 默认值
        end
        
        % 检测率计算
        if isfield(episode_results, 'attack_info')
            attack_success_rate = mean([episode_results.attack_info{:}]);
            metrics.detection_rate = 1 - attack_success_rate; % 检测率 = 1 - 攻击成功率
        else
            metrics.detection_rate = 0.7; % 默认检测率
        end
        
        % RADI指标
        if isfield(episode_results, 'avg_radi')
            metrics.avg_radi = mean(episode_results.avg_radi);
        else
            metrics.avg_radi = 1.0; % 默认RADI
        end
        
        % 奖励指标
        if isfield(episode_results, 'avg_defender_reward')
            metrics.avg_defender_reward = mean(episode_results.avg_defender_reward);
        else
            metrics.avg_defender_reward = 0;
        end
        
        if isfield(episode_results, 'avg_attacker_reward')
            metrics.avg_attacker_reward = episode_results.avg_attacker_reward;
        else
            metrics.avg_attacker_reward = 0;
        end
        
        % 系统安全级别
        metrics.security_level = metrics.detection_rate;
        
        % 资源利用率
        if isfield(config, 'total_resources')
            total_used = sum(metrics.resource_allocation);
            metrics.resource_utilization = total_used / config.total_resources;
        else
            metrics.resource_utilization = mean(metrics.resource_allocation) / 100;
        end
        
        % 尝试更新监控器
        if hasMethod(monitor, 'updateMetrics')
            % 如果monitor有updateMetrics方法
            monitor.updateMetrics(iteration, metrics);
        elseif hasMethod(monitor, 'update')
            % 如果monitor有update方法
            monitor.update(iteration, metrics);
        elseif isprop(monitor, 'metrics') || isfield(monitor, 'metrics')
            % 如果monitor有metrics属性，直接更新
            if isprop(monitor, 'metrics')
                monitor.metrics{iteration} = metrics;
            else
                monitor.metrics{iteration} = metrics;
            end
        else
            % 如果都没有，尝试添加字段
            try
                monitor.latest_metrics = metrics;
                monitor.last_update_iteration = iteration;
            catch
                % 如果都失败了，就什么都不做
                warning('无法更新性能监控器，monitor对象可能不支持更新操作');
            end
        end
        
        % 实时状态显示
        if isfield(config, 'performance') && isfield(config.performance, 'display_interval')
            display_interval = config.performance.display_interval;
        else
            display_interval = 50; % 默认显示间隔
        end
        
        if mod(iteration, display_interval) == 0
            if hasMethod(monitor, 'displayRealTimeStatus')
                monitor.displayRealTimeStatus(iteration);
            elseif hasMethod(monitor, 'display')
                monitor.display(iteration);
            else
                % 简单的状态显示
                displaySimpleStatus(iteration, metrics, config);
            end
        end
        
        % 记录关键事件
        if hasMethod(monitor, 'logEvent')
            if metrics.detection_rate > 0.9
                monitor.logEvent(iteration, 'HIGH_DETECTION_RATE', sprintf('检测率达到 %.1f%%', metrics.detection_rate * 100));
            end
            if metrics.resource_efficiency > 0.95
                monitor.logEvent(iteration, 'HIGH_EFFICIENCY', sprintf('资源效率达到 %.1f%%', metrics.resource_efficiency * 100));
            end
        end
        
    catch ME
        warning('更新性能监控器时出错 (迭代 %d): %s', iteration, ME.message);
        
        % 尝试简单的状态显示作为备用
        try
            if mod(iteration, 50) == 0
                fprintf('迭代 %d - 监控更新失败，但仿真继续运行\n', iteration);
            end
        catch
            % 即使简单显示也失败，就静默继续
        end
    end
end

function displaySimpleStatus(iteration, metrics, config)
    %% displaySimpleStatus - 简单的状态显示
    % 当monitor对象没有显示方法时使用
    
    fprintf('\n=== 迭代 %d 性能状态 ===\n', iteration);
    fprintf('检测率: %.1f%%\n', metrics.detection_rate * 100);
    fprintf('平均RADI: %.3f\n', metrics.avg_radi);
    fprintf('资源效率: %.1f%%\n', metrics.resource_efficiency * 100);
    fprintf('分配均衡: %.3f\n', metrics.allocation_balance);
    fprintf('资源利用率: %.1f%%\n', metrics.resource_utilization * 100);
    fprintf('防御者奖励: %.3f\n', metrics.avg_defender_reward);
    fprintf('攻击者奖励: %.3f\n', metrics.avg_attacker_reward);
    fprintf('========================\n');
end

function has_method = hasMethod(obj, method_name)
    %% hasMethod - 检查对象是否有指定方法
    try
        if isobject(obj)
            has_method = any(strcmp(methods(obj), method_name));
        else
            has_method = false;
        end
    catch
        has_method = false;
    end
end