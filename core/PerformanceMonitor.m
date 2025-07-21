%% PerformanceMonitor.m - 性能监控器类
% =========================================================================
% 描述: 监控和记录仿真过程中的各项性能指标。
%       该版本修复了属性初始化问题和损坏的 calculateRADI 方法。
% =========================================================================
classdef PerformanceMonitor < handle
    
    properties
        % === 基本参数 ===
        n_iterations          % 总迭代数
        n_agents             % 智能体数量
        config               % 配置参数
        
        % === 核心数据存储数组 (已在构造函数中初始化) ===
        defender_rewards     % 防御者奖励历史 [n_iterations x n_agents]
        attacker_rewards     % 攻击者奖励历史 [n_iterations x 1]
        detection_rates      % 检测率历史 [n_iterations x n_agents]
        
        % === RADI体系指标 ===
        radi_scores          % RADI得分历史
        resource_efficiency  % 资源效率
        allocation_balance   % 分配均衡
        
        % === 其他指标 ===
        episode_rewards      % 回合奖励历史
        training_loss        % 训练损失历史
        exploration_rates    % 探索率历史
        resource_allocations % 资源分配历史 (结构体)
        best_performance     % 最佳性能记录
        current_episode      % 当前回合数
        real_time_metrics    % 实时指标 (结构体)
        display_interval     % 显示间隔
    end
    
    methods
        %% 构造函数: 初始化所有属性
        function obj = PerformanceMonitor(n_iterations, n_agents, config)
            % --- 基本参数 ---
            obj.n_iterations = n_iterations;
            obj.n_agents = n_agents;
            obj.config = config;
            
            % --- 预分配核心数据数组，提高效率 ---
            obj.defender_rewards = NaN(n_iterations, n_agents);
            obj.attacker_rewards = NaN(n_iterations, 1);
            obj.detection_rates = NaN(n_iterations, n_agents);
            
            % --- 初始化其他指标数组 ---
            obj.radi_scores = [];
            obj.resource_efficiency = [];
            obj.allocation_balance = [];
            obj.episode_rewards = [];
            obj.training_loss = [];
            obj.exploration_rates = [];
            
            % --- 初始化结构体 ---
            obj.resource_allocations = struct('computation', [], 'bandwidth', [], 'sensors', [], 'scanning_freq', [], 'inspection_depth', []);
            obj.best_performance = struct('best_radi', Inf, 'best_efficiency', 0, 'best_balance', 0, 'best_radi_episode', 0);
            obj.real_time_metrics = struct();
            
            % --- 其他参数 ---
            obj.current_episode = 0;
            obj.display_interval = 50; % 每50个回合显示一次状态
        end
        
        %% 更新每个回合的指标
        function updateMetrics(obj, episode, metrics)
            % 更新性能指标，确保维度兼容性
            
            obj.current_episode = episode;
            
            % --- 确保 resource_allocation 是一个有效的行向量 ---
            if isfield(metrics, 'resource_allocation')
                resource_allocation = metrics.resource_allocation(:)'; % 强制转换为行向量
            else
                % 如果未提供，则使用默认的均匀分配
                resource_allocation = ones(1, 5) * 0.2; 
                fprintf('警告: 在回合 %d 中未提供 resource_allocation，使用默认值。\n', episode);
            end

            % --- 计算RADI得分 ---
            radi = obj.calculateRADI(resource_allocation);
            obj.radi_scores(end+1) = radi;

            % --- 更新资源效率 ---
            if isfield(metrics, 'resource_efficiency') && isscalar(metrics.resource_efficiency)
                obj.resource_efficiency(end+1) = metrics.resource_efficiency;
            else
                obj.resource_efficiency(end+1) = NaN; % 使用NaN填充缺失值
            end

            % --- 更新分配平衡度 ---
            if isfield(metrics, 'allocation_balance') && isscalar(metrics.allocation_balance)
                obj.allocation_balance(end+1) = metrics.allocation_balance;
            else
                obj.allocation_balance(end+1) = NaN; % 使用NaN填充缺失值
            end
            
            % --- 安全地更新资源分配的各个分量 ---
            if length(resource_allocation) == 5
                obj.resource_allocations.computation(end+1) = resource_allocation(1);
                obj.resource_allocations.bandwidth(end+1) = resource_allocation(2);
                obj.resource_allocations.sensors(end+1) = resource_allocation(3);
                obj.resource_allocations.scanning_freq(end+1) = resource_allocation(4);
                obj.resource_allocations.inspection_depth(end+1) = resource_allocation(5);
            else
                fprintf('警告: resource_allocation 维度不匹配 (期望5个元素)，在回合 %d 中跳过更新。\n', episode);
            end
            
            % --- 更新最佳性能记录 ---
            if radi < obj.best_performance.best_radi
                obj.best_performance.best_radi = radi;
                obj.best_performance.best_radi_episode = episode;
            end
            
            % --- 评估当前性能等级 ---
            obj.evaluateCurrentPerformance(metrics, radi);
        end
        
        %% 更新每次迭代的核心数据 (用于替代旧的 updateIteration)
        function updateIterationData(obj, iteration, episode_data)
            % 检查迭代索引是否在范围内
            if iteration > 0 && iteration <= obj.n_iterations
                if isfield(episode_data, 'avg_defender_reward')
                    obj.defender_rewards(iteration, :) = episode_data.avg_defender_reward;
                end
                if isfield(episode_data, 'avg_attacker_reward')
                    obj.attacker_rewards(iteration) = episode_data.avg_attacker_reward;
                end
                if isfield(episode_data, 'avg_detection_rate')
                    obj.detection_rates(iteration, :) = episode_data.avg_detection_rate;
                end
            else
                fprintf('警告: 迭代索引 %d 超出范围 [1, %d]。\n', iteration, obj.n_iterations);
            end
        end

        %% [已修复] 计算RADI得分
        function radi = calculateRADI(obj, resource_allocation)
            % 计算资源分配偏差指数 (RADI)
            % RADI = sum(weights .* abs(current_allocation - optimal_allocation))
            try
                % 从配置中获取最优分配和权重
                optimal = obj.config.radi.optimal_allocation(:)'; % 确保是行向量
                weights = obj.config.radi.weights(:)';           % 确保是行向量

                if length(resource_allocation) ~= length(optimal)
                    error('资源分配向量和最优分配向量的维度不匹配。');
                end

                % 计算加权绝对偏差
                deviation = abs(resource_allocation - optimal);
                radi = sum(weights .* deviation);

            catch ME
                fprintf('RADI 计算失败: %s\n', ME.message);
                fprintf('请确保 config.radi.optimal_allocation 和 config.radi.weights 已正确设置。\n');
                radi = NaN; % 返回NaN表示计算失败
            end
        end
        
        %% 评估当前性能等级
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
            
            % 更新实时指标结构体
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
        
        %% 显示实时状态
        function displayRealTimeStatus(obj, episode)
            if mod(episode, obj.display_interval) == 0 && episode > 0
                fprintf('\n===== 实时性能监控 [Episode %d] =====\n', episode);
                if isfield(obj.real_time_metrics, 'performance_level')
                    fprintf('  当前性能等级: %s\n', obj.real_time_metrics.performance_level);
                    fprintf('  当前RADI值: %.4f\n', obj.real_time_metrics.current_radi);
                    fprintf('  资源效率: %.2f%%\n', obj.real_time_metrics.current_efficiency * 100);
                    fprintf('  分配平衡度: %.2f%%\n', obj.real_time_metrics.current_balance * 100);
                end
                fprintf('\n  历史最佳:\n');
                fprintf('  最佳RADI: %.4f (在 Episode %d)\n', obj.best_performance.best_radi, obj.best_performance.best_radi_episode);
                
                if ~isempty(obj.resource_allocations.computation)
                    fprintf('\n  当前资源分配:\n');
                    fprintf('    计算资源: %.2f%%, 带宽: %.2f%%, 传感器: %.2f%%, 扫描频率: %.2f%%, 检查深度: %.2f%%\n', ...
                        obj.resource_allocations.computation(end) * 100, ...
                        obj.resource_allocations.bandwidth(end) * 100, ...
                        obj.resource_allocations.sensors(end) * 100, ...
                        obj.resource_allocations.scanning_freq(end) * 100, ...
                        obj.resource_allocations.inspection_depth(end) * 100);
                end
                fprintf('========================================\n\n');
            end
        end
        
        %% 生成最终摘要
        function summary = generateSummary(obj)
            if isempty(obj.radi_scores)
                summary = struct('message', '没有可用的性能数据');
                return;
            end
            
            % 使用最近100个回合或所有可用数据（如果少于100）
            window_size = min(100, length(obj.radi_scores));
            recent_idx = (length(obj.radi_scores) - window_size + 1):length(obj.radi_scores);
            
            summary = struct();
            summary.final_radi = nanmean(obj.radi_scores(recent_idx));
            summary.final_efficiency = nanmean(obj.resource_efficiency(recent_idx));
            summary.final_balance = nanmean(obj.allocation_balance(recent_idx));
            summary.final_allocation = [
                nanmean(obj.resource_allocations.computation(recent_idx)),
                nanmean(obj.resource_allocations.bandwidth(recent_idx)),
                nanmean(obj.resource_allocations.sensors(recent_idx)),
                nanmean(obj.resource_allocations.scanning_freq(recent_idx)),
                nanmean(obj.resource_allocations.inspection_depth(recent_idx))
            ];
            
            % 检查是否达到目标
            summary.target_achievement = struct();
            if isfield(obj.config, 'training') && isfield(obj.config.training, 'performance_target_radi')
                summary.target_achievement.radi_achieved = summary.final_radi <= obj.config.training.performance_target_radi;
                summary.target_achievement.target_radi = obj.config.training.performance_target_radi;
            else
                summary.target_achievement.radi_achieved = 'N/A';
            end
        end
        
        %% 生成改进建议
        function suggestions = generateImprovementSuggestions(obj)
            suggestions = {};
            if isempty(obj.real_time_metrics) || ~isfield(obj.real_time_metrics, 'current_radi')
                suggestions{1} = '数据不足，无法生成建议。';
                return;
            end

            current_radi = obj.real_time_metrics.current_radi;
            if current_radi > obj.config.radi.threshold_acceptable
                suggestions{end+1} = sprintf('当前RADI值(%.3f)偏高，建议优先优化资源分配策略。', current_radi);
            end

            if ~isempty(obj.resource_allocations.computation)
                current_allocation = [
                    obj.resource_allocations.computation(end), obj.resource_allocations.bandwidth(end),
                    obj.resource_allocations.sensors(end), obj.resource_allocations.scanning_freq(end),
                    obj.resource_allocations.inspection_depth(end)
                ];
                optimal = obj.config.radi.optimal_allocation;
                deviations = abs(current_allocation - optimal);
                [max_dev, max_idx] = max(deviations);

                if max_dev > 0.1 % 如果最大偏差超过10%
                    resource_names = {'计算资源', '带宽', '传感器', '扫描频率', '检查深度'};
                    suggestions{end+1} = sprintf('重点关注 "%s" 的分配，其当前值与最优值偏差最大 (偏差 %.1f%%)。', resource_names{max_idx}, max_dev * 100);
                end
            end

            if current_radi < obj.config.radi.threshold_excellent
                suggestions{end+1} = '当前资源分配表现优秀，可考虑在更复杂的攻击场景下进行压力测试以验证其鲁棒性。';
            end

            if isempty(suggestions)
                suggestions{1} = '当前资源分配策略表现良好，在可接受范围内，建议继续监控。';
            end
        end
        
        %% 获取所有结果数据
        function results = getResults(obj)
            results = struct();
            results.defender_rewards = obj.defender_rewards;
            results.attacker_rewards = obj.attacker_rewards;
            results.detection_rates = obj.detection_rates;
            results.radi_scores = obj.radi_scores;
            results.resource_efficiency = obj.resource_efficiency;
            results.allocation_balance = obj.allocation_balance;
            results.resource_allocations = obj.resource_allocations;
            results.summary = obj.generateSummary();
            results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            results.config = obj.config;
        end
    end
end

% 辅助函数可以在 classdef 文件末尾定义，但不能在 end 之后。
% 如果需要 softmax，建议将其作为类的静态方法或放在单独的 .m 文件中。
