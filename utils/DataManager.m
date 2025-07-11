%% DataManager.m - 数据管理器类
classdef DataManager
    methods (Static)
        function saveResults(results, config, agents)
            % 保存仿真结果
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
            agent_names = arrayfun(@(x) sprintf('Agent_%d', x), 1:results.n_agents, 'UniformOutput', false);
            % RADI数据
            if isfield(results, 'radi')
                radi_table = array2table(results.radi', 'VariableNames', agent_names);
                radi_table.Iteration = (1:results.n_iterations)';
                radi_table = radi_table(:, ['Iteration', agent_names]);
                writetable(radi_table, sprintf('results/radi_%s.csv', timestamp));
            end
            % 资源效率数据
            if isfield(results, 'resource_efficiency')
                eff_table = array2table(results.resource_efficiency', 'VariableNames', agent_names);
                eff_table.Iteration = (1:results.n_iterations)';
                eff_table = eff_table(:, ['Iteration', agent_names]);
                writetable(eff_table, sprintf('results/resource_efficiency_%s.csv', timestamp));
            end
            % 分配均衡性数据
            if isfield(results, 'allocation_balance')
                bal_table = array2table(results.allocation_balance', 'VariableNames', agent_names);
                bal_table.Iteration = (1:results.n_iterations)';
                bal_table = bal_table(:, ['Iteration', agent_names]);
                writetable(bal_table, sprintf('results/allocation_balance_%s.csv', timestamp));
            end
            % 收敛性数据
            if isfield(results, 'convergence_metrics')
                conv_table = array2table(results.convergence_metrics', 'VariableNames', agent_names);
                conv_table.Iteration = (1:results.n_iterations)';
                conv_table = conv_table(:, ['Iteration', agent_names]);
                writetable(conv_table, sprintf('results/convergence_metrics_%s.csv', timestamp));
            end
            % 奖励数据
            if isfield(results, 'defender_rewards')
                reward_table = array2table(results.defender_rewards', 'VariableNames', agent_names);
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
            last_iters = max(1, results.n_iterations-99):results.n_iterations;
            for i = 1:results.n_agents
                summary.agent(i).final_radi = mean(results.radi(i, last_iters));
                summary.agent(i).final_resource_efficiency = mean(results.resource_efficiency(i, last_iters));
                summary.agent(i).final_allocation_balance = mean(results.allocation_balance(i, last_iters));
                if isfield(results, 'convergence_metrics')
                    summary.agent(i).final_convergence = mean(results.convergence_metrics(i, last_iters));
                else
                    summary.agent(i).final_convergence = NaN;
                end
                summary.agent(i).max_radi = max(results.radi(i, :));
                summary.agent(i).min_radi = min(results.radi(i, :));
            end
            summary.overall_best_radi = min([summary.agent.final_radi]);
            summary.overall_best_efficiency = max([summary.agent.final_resource_efficiency]);
            summary.overall_best_balance = max([summary.agent.final_allocation_balance]);
            summary.total_iterations = results.n_iterations;
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