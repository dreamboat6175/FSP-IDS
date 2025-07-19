%% DataManager.m - 数据管理器类
classdef DataManager
    methods (Static)
       function saveResults(results, config, agents)
            % 保存仿真结果
            
            % 如果没有提供agents参数，设为空结构
            if nargin < 3
                agents = struct();
            end
            
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            if ~exist('results', 'dir')
                mkdir('results');
            end
            filename = sprintf('results/fsp_results_%s.mat', timestamp);
            save_data.results = results;
            save_data.config = config;
            save_data.timestamp = now;
            save_data.matlab_version = version;
            save_data.policies = {};
            save_data.agent_names = {};
            
            if isfield(agents, 'defenders')
                for i = 1:length(agents.defenders)
                    save_data.policies{i} = agents.defenders{i}.getPolicy();
                    save_data.agent_names{i} = agents.defenders{i}.name;
                    save_data.agent_stats{i} = agents.defenders{i}.getStatistics();
                end
            end
            if isfield(agents, 'attacker')
                save_data.attacker_policy = agents.attacker.getPolicy();
                save_data.attacker_stats = agents.attacker.getStatistics();
            end
            save_data.summary = DataManager.calculateSummaryStats(results);
            save(filename, 'save_data', '-v7.3');
            fprintf('✓ 结果已保存: %s\n', filename);
            DataManager.exportToCSV(results, timestamp);
        end
        function data = loadResults(filename)
            if exist(filename, 'file')
                loaded = load(filename);
                data = loaded.save_data;
                fprintf('✓ 结果已加载: %s\n', filename);
            else
                error('文件不存在: %s', filename);
            end
        end
        function exportToCSV(results, timestamp)
            if ~exist('results', 'dir')
                mkdir('results');
            end
            n_agents = results.n_agents;
            agent_names = arrayfun(@(x) sprintf('Agent_%d', x), 1:n_agents, 'UniformOutput', false);
            % RADI数据
            if isfield(results, 'radi')
                radi_data = results.radi(1:n_agents, :)';
                radi_table = array2table(radi_data, 'VariableNames', agent_names);
                radi_table.Iteration = (1:results.n_iterations)';
                radi_table = radi_table(:, ['Iteration', agent_names]);
                writetable(radi_table, sprintf('results/radi_%s.csv', timestamp));
            end
            % 资源效率数据
            if isfield(results, 'resource_efficiency')
                eff_data = results.resource_efficiency(1:n_agents, :)';
                eff_table = array2table(eff_data, 'VariableNames', agent_names);
                eff_table.Iteration = (1:results.n_iterations)';
                eff_table = eff_table(:, ['Iteration', agent_names]);
                writetable(eff_table, sprintf('results/resource_efficiency_%s.csv', timestamp));
            end
            % 分配均衡性数据
            if isfield(results, 'allocation_balance')
                bal_data = results.allocation_balance(1:n_agents, :)';
                bal_table = array2table(bal_data, 'VariableNames', agent_names);
                bal_table.Iteration = (1:results.n_iterations)';
                bal_table = bal_table(:, ['Iteration', agent_names]);
                writetable(bal_table, sprintf('results/allocation_balance_%s.csv', timestamp));
            end
            % 收敛性数据
            if isfield(results, 'convergence_metrics')
                conv_data = results.convergence_metrics(1:n_agents, :)';
                conv_table = array2table(conv_data, 'VariableNames', agent_names);
                conv_table.Iteration = (1:results.n_iterations)';
                conv_table = conv_table(:, ['Iteration', agent_names]);
                writetable(conv_table, sprintf('results/convergence_metrics_%s.csv', timestamp));
            end
            % 奖励数据
            if isfield(results, 'defender_rewards')
                reward_data = results.defender_rewards(1:n_agents, :)';
                reward_table = array2table(reward_data, 'VariableNames', agent_names);
                reward_table.Iteration = (1:results.n_iterations)';
                reward_table = reward_table(:, ['Iteration', agent_names]);
                writetable(reward_table, sprintf('results/defender_rewards_%s.csv', timestamp));
            end
            % 汇总统计
            summary_data = DataManager.calculateSummaryStats(results);
            summary_table = struct2table(summary_data);
            writetable(summary_table, sprintf('results/summary_stats_%s.csv', timestamp));
            fprintf('✓ CSV文件已导出到results目录\n');
        end
        function summary = calculateSummaryStats(results)
            % 获取迭代次数 - 从现有数据推断
            if isfield(results, 'n_iterations')
                n_iterations = results.n_iterations;
            elseif isfield(results, 'defender_rewards')
                n_iterations = size(results.defender_rewards, 1);
            elseif isfield(results, 'attacker_rewards')
                n_iterations = length(results.attacker_rewards);
            else
                n_iterations = 100; % 默认值
            end
            
            last_iters = max(1, n_iterations-99):n_iterations;
            
            % 检查是否有数据
            if ~isfield(results, 'defender_rewards') || isempty(results.defender_rewards)
                % 没有数据，返回空结构
                summary = struct();
                summary.total_iterations = n_iterations;
                summary.overall_best_radi = NaN;
                summary.overall_best_efficiency = NaN;
                summary.overall_best_balance = NaN;
                return;
            end
            
            % shape检查
            [n_iterations_actual, n_agents] = size(results.defender_rewards);
            
            for i = 1:n_agents
                % 越界保护
                valid_iters = last_iters(last_iters <= n_iterations_actual);
                if isempty(valid_iters)
                    summary.agent(i).final_radi = NaN;
                    summary.agent(i).final_resource_efficiency = NaN;
                    summary.agent(i).final_allocation_balance = NaN;
                    summary.agent(i).final_convergence = NaN;
                    summary.agent(i).max_radi = NaN;
                    summary.agent(i).min_radi = NaN;
                    continue;
                end
                
                % 使用defender_rewards作为默认指标
                summary.agent(i).final_radi = mean(results.defender_rewards(valid_iters, i));
                summary.agent(i).max_radi = max(results.defender_rewards(:, i));
                summary.agent(i).min_radi = min(results.defender_rewards(:, i));
                
                % 如果有其他字段，使用它们
                if isfield(results, 'resource_efficiency')
                    summary.agent(i).final_resource_efficiency = mean(results.resource_efficiency(valid_iters, i));
                else
                    summary.agent(i).final_resource_efficiency = NaN;
                end
                
                if isfield(results, 'allocation_balance')
                    summary.agent(i).final_allocation_balance = mean(results.allocation_balance(valid_iters, i));
                else
                    summary.agent(i).final_allocation_balance = NaN;
                end
                
                if isfield(results, 'convergence_metrics')
                    summary.agent(i).final_convergence = mean(results.convergence_metrics(valid_iters, i));
                else
                    summary.agent(i).final_convergence = NaN;
                end
            end
            
            % 计算总体最优值
            summary.overall_best_radi = min([summary.agent.final_radi]);
            summary.overall_best_efficiency = max([summary.agent.final_resource_efficiency]);
            summary.overall_best_balance = max([summary.agent.final_allocation_balance]);
            summary.total_iterations = n_iterations_actual;
        end
        function mergeResults(filenames)
            if isempty(filenames)
                error('需要提供文件名列表');
            end
            merged_data = [];
            for i = 1:length(filenames)
                data = DataManager.loadResults(filenames{i});
                if isempty(merged_data)
                    merged_data = data;
                    merged_data.results.radi_all = data.results.radi;
                    merged_data.results.resource_efficiency_all = data.results.resource_efficiency;
                    merged_data.results.allocation_balance_all = data.results.allocation_balance;
                else
                    merged_data.results.radi_all = cat(3, merged_data.results.radi_all, data.results.radi);
                    merged_data.results.resource_efficiency_all = cat(3, merged_data.results.resource_efficiency_all, data.results.resource_efficiency);
                    merged_data.results.allocation_balance_all = cat(3, merged_data.results.allocation_balance_all, data.results.allocation_balance);
                end
            end
            merged_data.results.radi_mean = mean(merged_data.results.radi_all, 3);
            merged_data.results.radi_std = std(merged_data.results.radi_all, 0, 3);
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            save(sprintf('results/merged_results_%s.mat', timestamp), 'merged_data');
            fprintf('✓ 合并结果已保存\n');
        end
    end
end