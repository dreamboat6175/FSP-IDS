function results = recordIterationResults(results, episode_results, iteration)
    %% recordIterationResults - 记录单次迭代的结果
    % 输入:
    %   results - 结果结构体
    %   episode_results - 当前迭代的episode结果
    %   iteration - 当前迭代编号
    % 输出:
    %   results - 更新后的结果结构体
    
    try
        % 记录主要性能指标
        if isfield(episode_results, 'avg_radi')
            results.radi(:, iteration) = episode_results.avg_radi(:);
        end
        
        if isfield(episode_results, 'avg_efficiency')
            results.resource_efficiency(:, iteration) = episode_results.avg_efficiency(:);
        end
        
        if isfield(episode_results, 'avg_balance')
            results.allocation_balance(:, iteration) = episode_results.avg_balance(:);
        end
        
        % 记录奖励信息
        if isfield(episode_results, 'avg_defender_reward')
            results.defender_rewards(:, iteration) = episode_results.avg_defender_reward(:);
        end
        
        if isfield(episode_results, 'avg_attacker_reward')
            results.attacker_rewards(1, iteration) = episode_results.avg_attacker_reward;
        end
        
        % 记录成功率和检测率
        if isfield(episode_results, 'attack_info')
            attack_success_rate = mean([episode_results.attack_info{:}]);
            results.success_rates(:, iteration) = 1 - attack_success_rate; % 成功率 = 1 - 攻击成功率
            
            % 如果有历史字段，也更新它们
            if isfield(results, 'success_rate_history')
                results.success_rate_history(iteration) = 1 - attack_success_rate;
            end
            if isfield(results, 'detection_rate_history')
                results.detection_rate_history(iteration) = 1 - attack_success_rate;
            end
        end
        
        % 记录策略信息
        if isfield(episode_results, 'attacker_strategy')
            if isfield(results, 'attacker_strategy_history')
                if size(results.attacker_strategy_history, 2) >= length(episode_results.attacker_strategy)
                    results.attacker_strategy_history(iteration, 1:length(episode_results.attacker_strategy)) = episode_results.attacker_strategy;
                end
            end
        end
        
        if isfield(episode_results, 'defender_strategies')
            if isfield(results, 'defender_strategy_history')
                for i = 1:length(episode_results.defender_strategies)
                    if i <= length(results.defender_strategy_history)
                        strategy = episode_results.defender_strategies{i};
                        if size(results.defender_strategy_history{i}, 2) >= length(strategy)
                            results.defender_strategy_history{i}(iteration, 1:length(strategy)) = strategy;
                        end
                    end
                end
            end
        end
        
        % 记录资源分配信息
        if isfield(episode_results, 'avg_resource_allocation')
            if ~isfield(results, 'resource_allocation_history')
                [n_agents, n_stations] = size(episode_results.avg_resource_allocation);
                results.resource_allocation_history = zeros(n_agents, n_stations, results.n_iterations);
            end
            results.resource_allocation_history(:, :, iteration) = episode_results.avg_resource_allocation;
        end
        
        % 记录累积奖励
        if isfield(results, 'cumulative_rewards')
            if iteration == 1
                results.cumulative_rewards(:, iteration) = results.defender_rewards(:, iteration);
            else
                results.cumulative_rewards(:, iteration) = results.cumulative_rewards(:, iteration-1) + results.defender_rewards(:, iteration);
            end
        end
        
        % 计算收敛性指标
        if iteration > 1 && isfield(results, 'convergence_metrics')
            % RADI变化率作为收敛指标
            radi_change = abs(results.radi(:, iteration) - results.radi(:, iteration-1));
            results.convergence_metrics(:, iteration) = radi_change;
        elseif iteration == 1 && isfield(results, 'convergence_metrics')
            results.convergence_metrics(:, iteration) = ones(size(results.radi, 1), 1);
        end
        
        % 计算策略多样性（如果有策略历史）
        if iteration > 5 && isfield(results, 'strategy_diversity')
            for agent_idx = 1:results.n_agents
                if isfield(results, 'defender_strategy_history') && agent_idx <= length(results.defender_strategy_history)
                    recent_strategies = results.defender_strategy_history{agent_idx}(max(1, iteration-4):iteration, :);
                    diversity = std(recent_strategies(:));
                    results.strategy_diversity(agent_idx, iteration) = diversity;
                end
            end
        end
        
        % 记录系统级指标
        if isfield(results, 'system_security_level')
            % 系统安全级别 = 平均检测率
            avg_detection_rate = mean(results.success_rates(:, iteration));
            results.system_security_level(iteration) = avg_detection_rate;
        end
        
        if isfield(results, 'total_resource_consumption')
            % 总资源消耗
            if isfield(episode_results, 'avg_resource_allocation')
                total_consumption = sum(episode_results.avg_resource_allocation(:));
                results.total_resource_consumption(iteration) = total_consumption;
            end
        end
        
        if isfield(results, 'network_coverage')
            % 网络覆盖率（简化计算）
            if isfield(episode_results, 'avg_resource_allocation')
                non_zero_stations = sum(episode_results.avg_resource_allocation > 0, 2);
                avg_coverage = mean(non_zero_stations) / size(episode_results.avg_resource_allocation, 2);
                results.network_coverage(iteration) = avg_coverage;
            end
        end
        
        % 更新时间记录
        if isfield(results, 'iteration_times')
            if iteration == 1
                results.iteration_times(iteration) = toc; % 假设在主程序中已经tic
            else
                results.iteration_times(iteration) = toc;
            end
        end
        
        % 更新状态
        results.status = sprintf('iteration_%d_completed', iteration);
        
    catch ME
        warning('记录迭代结果时出错 (迭代 %d): %s', iteration, ME.message);
        fprintf('错误详情: %s\n', ME.getReport());
        
        % 至少确保基本字段被填充
        if isfield(episode_results, 'avg_radi') && size(results.radi, 2) >= iteration
            results.radi(:, iteration) = episode_results.avg_radi(:);
        end
        if isfield(episode_results, 'avg_defender_reward') && size(results.defender_rewards, 2) >= iteration
            results.defender_rewards(:, iteration) = episode_results.avg_defender_reward(:);
        end
    end
    
    % 验证数据完整性
    if mod(iteration, 50) == 0
        fprintf('✓ 迭代 %d 结果已记录\n', iteration);
    end
end