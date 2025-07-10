%% PerformanceMonitor.m - 性能监控器类
% =========================================================================
% 描述: 监控和记录仿真过程中的各项性能指标
% =========================================================================

classdef PerformanceMonitor < handle
    
    properties
        % 基本参数
        n_iterations          % 总迭代数
        n_agents             % 智能体数量
        
        % 性能指标
        detection_rates      % 检测率记录
        attack_success_rates % 攻击成功率
        defender_rewards     % 防御者奖励
        attacker_rewards     % 攻击者奖励
        resource_utilization % 资源利用率
        convergence_metrics  % 收敛度量
        exploitability      % 可利用性
        nash_gap            % 纳什均衡差距
        
        % 计算性能
        computation_time    % 计算时间
        memory_usage       % 内存使用
        iteration_time     % 每次迭代时间
        
        % 攻击分析
        attack_type_stats   % 攻击类型统计
        target_distribution % 目标分布
        
        % 策略分析
        strategy_diversity  % 策略多样性
        policy_stability   % 策略稳定性
        
        % 实时数据
        current_iter       % 当前迭代
        start_time        % 开始时间
    end
    
    methods
        function obj = PerformanceMonitor(n_iterations, n_agents)
            % 构造函数
            obj.n_iterations = n_iterations;
            obj.n_agents = n_agents;
            
            % 初始化性能指标数组
            obj.detection_rates = zeros(n_agents, n_iterations);
            obj.attack_success_rates = zeros(n_agents, n_iterations);
            obj.defender_rewards = zeros(n_agents, n_iterations);
            obj.attacker_rewards = zeros(1, n_iterations);
            obj.resource_utilization = zeros(n_agents, n_iterations);
            obj.convergence_metrics = zeros(n_agents, n_iterations);
            obj.exploitability = zeros(n_agents, n_iterations);
            obj.nash_gap = zeros(n_agents, n_iterations);
            
            % 计算性能
            obj.computation_time = zeros(1, n_iterations);
            obj.memory_usage = zeros(1, n_iterations);
            obj.iteration_time = zeros(1, n_iterations);
            
            % 攻击统计（假设最多10种攻击类型）
            obj.attack_type_stats = zeros(10, n_iterations);
            obj.target_distribution = [];
            
            % 策略分析
            obj.strategy_diversity = zeros(n_agents, n_iterations);
            obj.policy_stability = zeros(n_agents, n_iterations);
            
            % 记录开始时间
            obj.start_time = tic;
            obj.current_iter = 0;
        end
        
        function update(obj, iter, episode_results, defender_agents, attacker_agent, env)
            % 更新监控指标
            
            obj.current_iter = iter;
            tic_update = tic;
            
            % 基本性能指标
            for i = 1:obj.n_agents
                obj.detection_rates(i, iter) = episode_results.avg_detection_rate(i);
                obj.attack_success_rates(i, iter) = 1 - obj.detection_rates(i, iter);
                obj.defender_rewards(i, iter) = episode_results.avg_defender_reward(i);
            end
            obj.attacker_rewards(iter) = episode_results.avg_attacker_reward;
            
            % 资源利用率
            for i = 1:obj.n_agents
                obj.resource_utilization(i, iter) = ...
                    obj.calculateResourceUtilization(defender_agents{i}, env);
            end
            
            % 收敛性指标
            if iter > 10
                for i = 1:obj.n_agents
                    obj.convergence_metrics(i, iter) = ...
                        std(obj.detection_rates(i, max(1,iter-9):iter));
                end
            end
            
            % 策略分析（每10次迭代计算一次）
            if mod(iter, 10) == 0
                for i = 1:obj.n_agents
                    obj.exploitability(i, iter) = ...
                        obj.calculateExploitability(defender_agents{i}, attacker_agent);
                    obj.nash_gap(i, iter) = ...
                        obj.calculateNashGap(defender_agents{i}, env);
                    obj.strategy_diversity(i, iter) = ...
                        obj.calculateStrategyDiversity(defender_agents{i});
                end
            end
            
            % 攻击模式分析
            if isfield(episode_results, 'attack_info')
                obj.updateAttackStatistics(iter, episode_results.attack_info);
            end
            
            % 策略稳定性
            if iter > 1
                for i = 1:obj.n_agents
                    obj.policy_stability(i, iter) = ...
                        obj.calculatePolicyStability(defender_agents{i}, iter);
                end
            end
            
            % 系统性能
            obj.computation_time(iter) = toc(tic_update);
            obj.memory_usage(iter) = obj.getMemoryUsage();
            
            if isfield(episode_results, 'iteration_time')
                obj.iteration_time(iter) = episode_results.iteration_time;
            end
        end
        
        function utilization = calculateResourceUtilization(obj, agent, env)
            % 计算资源利用效率
            policy = agent.getPolicy();
            
            % 计算平均Q值利用率
            avg_q = mean(policy(:));
            max_q = max(policy(:));
            
            if max_q > 0
                utilization = avg_q / max_q;
            else
                utilization = 0;
            end
            
            % 考虑动作分布的均衡性
            action_probs = mean(softmax(policy / 0.1, 2), 1);
            entropy = -sum(action_probs .* log(action_probs + 1e-10));
            max_entropy = log(size(policy, 2));
            
            % 综合指标
            utilization = 0.7 * utilization + 0.3 * (entropy / max_entropy);
        end
        
        function exploitability = calculateExploitability(obj, defender, attacker)
            % 计算防御策略的可利用性
            defender_policy = defender.getPolicy();
            attacker_policy = attacker.getPolicy();
            
            % 计算攻击者针对防御者的最佳响应值
            best_response_values = max(attacker_policy, [], 2);
            current_values = mean(defender_policy, 2);
            
            exploitability = mean(best_response_values - current_values);
        end
        
        function nash_gap = calculateNashGap(obj, agent, env)
            % 计算纳什均衡差距
            policy = agent.getPolicy();
            
            % 简化的纳什差距计算
            best_response = max(policy, [], 2);
            avg_response = mean(policy, 2);
            
            nash_gap = mean(abs(best_response - avg_response));
        end
        
        function diversity = calculateStrategyDiversity(obj, agent)
            % 计算策略多样性
            policy = agent.getPolicy();
            
            % 基于熵的多样性度量
            policy_probs = softmax(policy / 0.1, 2);
            entropy_values = -sum(policy_probs .* log(policy_probs + 1e-10), 2);
            
            diversity = mean(entropy_values);
        end
        
        function stability = calculatePolicyStability(obj, agent, iter)
            % 计算策略稳定性
            if iter < 2
                stability = 1;
                return;
            end
            
            % 获取最近的策略变化
            recent_rewards = obj.defender_rewards(agent == agent, max(1, iter-9):iter);
            
            if length(recent_rewards) > 1
                stability = 1 / (1 + std(recent_rewards));
            else
                stability = 1;
            end
        end
        
        function updateAttackStatistics(obj, iter, attack_info)
            % 更新攻击统计信息
            if isempty(attack_info)
                return;
            end
            
            % 统计攻击类型分布
            attack_types = zeros(10, 1);
            for i = 1:length(attack_info)
                if ~isempty(attack_info{i}) && isfield(attack_info{i}, 'attack_type')
                    type_idx = attack_info{i}.attack_type;
                    if type_idx <= 10
                        attack_types(type_idx) = attack_types(type_idx) + 1;
                    end
                end
            end
            
            obj.attack_type_stats(:, iter) = attack_types / length(attack_info);
        end
        
        function memory_mb = getMemoryUsage(obj)
            % 获取当前内存使用量
            user = memory;
            memory_mb = user.MemUsedMATLAB / 1e6;
        end
        
        function results = getResults(obj)
            % 获取所有监控结果
            results.detection_rates = obj.detection_rates;
            results.attack_success_rates = obj.attack_success_rates;
            results.defender_rewards = obj.defender_rewards;
            results.attacker_rewards = obj.attacker_rewards;
            results.resource_utilization = obj.resource_utilization;
            results.convergence_metrics = obj.convergence_metrics;
            results.exploitability = obj.exploitability;
            results.nash_gap = obj.nash_gap;
            results.computation_time = obj.computation_time;
            results.memory_usage = obj.memory_usage;
            results.iteration_time = obj.iteration_time;
            results.attack_type_stats = obj.attack_type_stats;
            results.strategy_diversity = obj.strategy_diversity;
            results.policy_stability = obj.policy_stability;
            results.n_iterations = obj.n_iterations;
            results.n_agents = obj.n_agents;
        end
        
        function summary = getSummary(obj)
            % 获取性能摘要
            iter = obj.current_iter;
            if iter == 0
                summary = '尚未开始监控';
                return;
            end
            
            summary = sprintf('\n=== 性能监控摘要 (迭代 %d) ===\n', iter);
            
            % 检测率
            summary = [summary, sprintf('平均检测率:\n')];
            for i = 1:obj.n_agents
                recent_rate = mean(obj.detection_rates(i, max(1,iter-99):iter));
                summary = [summary, sprintf('  智能体%d: %.2f%%\n', i, recent_rate * 100)];
            end
            
            % 资源效率
            summary = [summary, sprintf('\n平均资源利用率:\n')];
            for i = 1:obj.n_agents
                recent_util = mean(obj.resource_utilization(i, max(1,iter-99):iter));
                summary = [summary, sprintf('  智能体%d: %.2f%%\n', i, recent_util * 100)];
            end
            
            % 收敛性
            summary = [summary, sprintf('\n策略稳定性 (越小越稳定):\n')];
            for i = 1:obj.n_agents
                if iter > 10
                    summary = [summary, sprintf('  智能体%d: %.4f\n', i, obj.convergence_metrics(i, iter))];
                end
            end
            
            % 计算性能
            avg_comp_time = mean(obj.computation_time(max(1,iter-99):iter));
            avg_memory = mean(obj.memory_usage(max(1,iter-99):iter));
            summary = [summary, sprintf('\n系统性能:\n')];
            summary = [summary, sprintf('  平均计算时间: %.3f秒/迭代\n', avg_comp_time)];
            summary = [summary, sprintf('  平均内存使用: %.1fMB\n', avg_memory)];
            
            summary = [summary, '==============================\n'];
        end
        
        function plotRealTimeMetrics(obj)
            % 实时绘制关键指标
            if obj.current_iter < 2
                return;
            end
            
            figure(999);
            clf;
            
            % 检测率趋势
            subplot(2, 2, 1);
            hold on;
            colors = lines(obj.n_agents);
            for i = 1:obj.n_agents
                plot(1:obj.current_iter, obj.detection_rates(i, 1:obj.current_iter), ...
                     'Color', colors(i,:), 'LineWidth', 1.5);
            end
            xlabel('迭代');
            ylabel('检测率');
            title('检测率实时监控');
            grid on;
            ylim([0, 1]);
            
            % 奖励趋势
            subplot(2, 2, 2);
            plot(1:obj.current_iter, obj.attacker_rewards(1:obj.current_iter), 'r-', 'LineWidth', 1.5);
            hold on;
            for i = 1:obj.n_agents
                plot(1:obj.current_iter, obj.defender_rewards(i, 1:obj.current_iter), ...
                     'Color', colors(i,:), 'LineWidth', 1);
            end
            xlabel('迭代');
            ylabel('平均奖励');
            title('奖励趋势');
            grid on;
            
            % 收敛性
            subplot(2, 2, 3);
            valid_idx = 11:obj.current_iter;
            if ~isempty(valid_idx)
                for i = 1:obj.n_agents
                    semilogy(valid_idx, obj.convergence_metrics(i, valid_idx), ...
                            'Color', colors(i,:), 'LineWidth', 1.5);
                    hold on;
                end
                xlabel('迭代');
                ylabel('标准差 (对数尺度)');
                title('策略收敛性');
                grid on;
            end
            
            % 系统性能
            subplot(2, 2, 4);
            yyaxis left;
            plot(1:obj.current_iter, obj.computation_time(1:obj.current_iter), 'b-', 'LineWidth', 1.5);
            ylabel('计算时间 (秒)');
            
            yyaxis right;
            plot(1:obj.current_iter, obj.memory_usage(1:obj.current_iter), 'r-', 'LineWidth', 1.5);
            ylabel('内存使用 (MB)');
            
            xlabel('迭代');
            title('系统资源使用');
            grid on;
            
            drawnow;
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