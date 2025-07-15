%% PerformanceMonitor.m - 性能监控器类
% =========================================================================
% 描述: 监控和记录仿真过程中的各项性能指标
% =========================================================================

classdef PerformanceMonitor < handle
    
    properties
        % 基本参数
        n_iterations          % 总迭代数
        n_agents             % 智能体数量
        
        % RADI体系指标
        radi_scores
        resource_efficiency
        allocation_balance
        episode_rewards
        training_loss
        exploration_rates
        resource_allocations
        best_performance
        current_episode
        real_time_metrics
        display_interval
        config
    end
    
    methods
        function obj = PerformanceMonitor(n_iterations, n_agents, config)
            obj.n_iterations = n_iterations;
            obj.n_agents = n_agents;
            obj.radi_scores = [];
            obj.resource_efficiency = [];
            obj.allocation_balance = [];
            obj.episode_rewards = [];
            obj.training_loss = [];
            obj.exploration_rates = [];
            obj.resource_allocations = struct('computation', [], 'bandwidth', [], 'sensors', [], 'scanning_freq', [], 'inspection_depth', []);
            obj.best_performance = struct('best_radi', Inf, 'best_efficiency', 0, 'best_balance', 0, 'best_radi_episode', 0);
            obj.current_episode = 0;
            obj.real_time_metrics = struct();
            obj.display_interval = 50;
            obj.config = config;
        end
        
        function updateMetrics(obj, episode, metrics)
            obj.current_episode = episode;
            radi = obj.calculateRADI(metrics.resource_allocation);
            obj.radi_scores(end+1) = radi;
            if isfield(metrics, 'resource_efficiency')
                obj.resource_efficiency(end+1) = metrics.resource_efficiency;
            else
                obj.resource_efficiency(end+1) = NaN;
            end
            if isfield(metrics, 'allocation_balance')
                obj.allocation_balance(end+1) = metrics.allocation_balance;
            else
                obj.allocation_balance(end+1) = NaN;
            end
            obj.resource_allocations.computation(end+1) = metrics.resource_allocation(1);
            obj.resource_allocations.bandwidth(end+1) = metrics.resource_allocation(2);
            obj.resource_allocations.sensors(end+1) = metrics.resource_allocation(3);
            obj.resource_allocations.scanning_freq(end+1) = metrics.resource_allocation(4);
            obj.resource_allocations.inspection_depth(end+1) = metrics.resource_allocation(5);
            if radi < obj.best_performance.best_radi
                obj.best_performance.best_radi = radi;
                obj.best_performance.best_radi_episode = episode;
            end
            obj.evaluateCurrentPerformance(metrics, radi);
        end
        
        function update(obj, episode, metrics, varargin)
            % 兼容旧接口，直接调用 updateMetrics
            obj.updateMetrics(episode, metrics);
        end
        
        function radi = calculateRADI(obj, resource_allocation)
            optimal = obj.config.radi.optimal_allocation;
            weights = [obj.config.radi.weight_computation, obj.config.radi.weight_bandwidth, obj.config.radi.weight_sensors, obj.config.radi.weight_scanning, obj.config.radi.weight_inspection];
            % === 修正：自动对齐向量长度 ===
            len = min([length(resource_allocation), length(optimal), length(weights)]);
            resource_allocation = resource_allocation(1:len);
            optimal = optimal(1:len);
            weights = weights(1:len);
            deviation = abs(resource_allocation - optimal);
            radi = sum(weights .* deviation);
        end
        
        function performance_level = evaluateCurrentPerformance(obj, metrics, radi)
            if radi <= obj.config.radi.threshold_excellent
                performance_level = 'excellent';
            elseif radi <= obj.config.radi.threshold_good
                performance_level = 'good';
            elseif radi <= obj.config.radi.threshold_acceptable
                performance_level = 'acceptable';
            else
                performance_level = 'needs_improvement';
            end
            obj.real_time_metrics.performance_level = performance_level;
            obj.real_time_metrics.current_radi = radi;
            if isfield(metrics, 'resource_efficiency')
                obj.real_time_metrics.current_efficiency = metrics.resource_efficiency;
            else
                obj.real_time_metrics.current_efficiency = NaN;
            end
            if isfield(metrics, 'allocation_balance')
                obj.real_time_metrics.current_balance = metrics.allocation_balance;
            else
                obj.real_time_metrics.current_balance = NaN;
            end
        end
        
        function displayRealTimeStatus(obj, episode)
            if mod(episode, obj.display_interval) == 0
                fprintf('\n=== 实时性能监控 [Episode %d] ===\n', episode);
                if ~isempty(obj.real_time_metrics)
                    fprintf('当前性能等级: %s\n', obj.real_time_metrics.performance_level);
                    fprintf('当前RADI值: %.3f\n', obj.real_time_metrics.current_radi);
                    fprintf('资源效率: %.2f%%\n', obj.real_time_metrics.current_efficiency * 100);
                    fprintf('分配平衡度: %.2f%%\n', obj.real_time_metrics.current_balance * 100);
                end
                fprintf('\n历史最佳:\n');
                fprintf('最佳RADI: %.3f (Episode %d)\n', obj.best_performance.best_radi, obj.best_performance.best_radi_episode);
                if ~isempty(obj.resource_allocations.computation)
                    fprintf('\n当前资源分配:\n');
                    fprintf('  计算资源: %.2f%%\n', obj.resource_allocations.computation(end) * 100);
                    fprintf('  带宽资源: %.2f%%\n', obj.resource_allocations.bandwidth(end) * 100);
                    fprintf('  传感器: %.2f%%\n', obj.resource_allocations.sensors(end) * 100);
                    fprintf('  扫描频率: %.2f%%\n', obj.resource_allocations.scanning_freq(end) * 100);
                    fprintf('  检查深度: %.2f%%\n', obj.resource_allocations.inspection_depth(end) * 100);
                end
                fprintf('================================\n\n');
            end
        end
        
        function summary = generateSummary(obj)
            if isempty(obj.radi_scores)
                summary = struct('message', '暂无性能数据');
                return;
            end
            window_size = min(100, length(obj.radi_scores));
            recent_idx = length(obj.radi_scores) - window_size + 1;
            recent_idx = max(1, recent_idx);
            summary = struct();
            summary.final_radi = mean(obj.radi_scores(recent_idx:end));
            summary.final_efficiency = mean(obj.resource_efficiency(recent_idx:end));
            summary.final_balance = mean(obj.allocation_balance(recent_idx:end));
            summary.final_allocation = [
                mean(obj.resource_allocations.computation(recent_idx:end)),
                mean(obj.resource_allocations.bandwidth(recent_idx:end)),
                mean(obj.resource_allocations.sensors(recent_idx:end)),
                mean(obj.resource_allocations.scanning_freq(recent_idx:end)),
                mean(obj.resource_allocations.inspection_depth(recent_idx:end))
            ];
            summary.target_achievement = struct();
            if isfield(obj.config, 'training') && isfield(obj.config.training, 'performance_target_radi')
                summary.target_achievement.radi_achieved = summary.final_radi <= obj.config.training.performance_target_radi;
            else
                summary.target_achievement.radi_achieved = NaN;
            end
        end
        
        function suggestions = generateImprovementSuggestions(obj)
            suggestions = {};
            if ~isempty(obj.real_time_metrics)
                current_radi = obj.real_time_metrics.current_radi;
                if current_radi > 0.3
                    suggestions{end+1} = '建议优化资源分配策略，当前RADI值偏高';
                end
                if ~isempty(obj.resource_allocations.computation)
                    current_allocation = [
                        obj.resource_allocations.computation(end),
                        obj.resource_allocations.bandwidth(end),
                        obj.resource_allocations.sensors(end),
                        obj.resource_allocations.scanning_freq(end),
                        obj.resource_allocations.inspection_depth(end)
                    ];
                    optimal = obj.config.radi.optimal_allocation;
                    deviations = abs(current_allocation - optimal);
                    [max_dev, max_idx] = max(deviations);
                    if max_dev > 0.1
                        resource_names = {'计算资源', '带宽', '传感器', '扫描频率', '检查深度'};
                        suggestions{end+1} = sprintf('调整%s分配，当前偏差%.1f%%', resource_names{max_idx}, max_dev * 100);
                    end
                end
                if current_radi < 0.1
                    suggestions{end+1} = '资源分配表现优秀，可以考虑在更复杂的场景中测试';
                end
            end
            if isempty(suggestions)
                suggestions{1} = '当前资源分配策略表现良好，继续保持';
            end
        end

        function results = getResults(obj)
            % 返回所有关键监控指标
            results = struct();
            results.radi_scores = obj.radi_scores;
            results.resource_efficiency = obj.resource_efficiency;
            results.allocation_balance = obj.allocation_balance;
            results.episode_rewards = obj.episode_rewards;
            results.training_loss = obj.training_loss;
            results.exploration_rates = obj.exploration_rates;
            results.resource_allocations = obj.resource_allocations;
            results.best_performance = obj.best_performance;
            results.n_iterations = obj.n_iterations;
            results.n_agents = obj.n_agents;
        end
    end
end

%% 辅助函数
function p = softmax(x, dim)
    % Softmax函数
    if nargin < 2
        dim = 2;
    end
    ex = exp(x - max(x, [], dim));
    p = ex ./ sum(ex, dim);
end