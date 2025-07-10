%% DataManager.m - 数据管理器类
classdef DataManager
    methods (Static)
        function saveResults(results, config, agents)
            % 保存仿真结果
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            
            % 确保结果目录存在
            if ~exist('results', 'dir')
                mkdir('results');
            end
            
            filename = sprintf('results/fsp_results_%s.mat', timestamp);
            
            % 整理数据
            save_data.results = results;
            save_data.config = config;
            save_data.timestamp = now;
            save_data.matlab_version = version;
            
            % 保存智能体策略
            save_data.policies = {};
            save_data.agent_names = {};
            
            if isfield(agents, 'defenders')
                for i = 1:length(agents.defenders)
                    save_data.policies{i} = agents.defenders{i}.getPolicy();
                    save_data.agent_names{i} = agents.defenders{i}.name;
                    save_data.agent_stats{i} = agents.defenders{i}.getStatistics();
                end
            end
            
            % 保存攻击者策略
            if isfield(agents, 'attacker')
                save_data.attacker_policy = agents.attacker.getPolicy();
                save_data.attacker_stats = agents.attacker.getStatistics();
            end
            
            % 计算汇总统计
            save_data.summary = DataManager.calculateSummaryStats(results);
            
            % 保存文件
            save(filename, 'save_data', '-v7.3');
            fprintf('✓ 结果已保存: %s\n', filename);
            
            % 导出关键数据为CSV
            DataManager.exportToCSV(results, timestamp);
        end
        
        function data = loadResults(filename)
            % 加载仿真结果
            if exist(filename, 'file')
                loaded = load(filename);
                data = loaded.save_data;
                fprintf('✓ 结果已加载: %s\n', filename);
            else
                error('文件不存在: %s', filename);
            end
        end
        
        function exportToCSV(results, timestamp)
            % 导出关键数据为CSV格式
            
            % 确保结果目录存在
            if ~exist('results', 'dir')
                mkdir('results');
            end
            
            % 检测率数据
            agent_names = arrayfun(@(x) sprintf('Agent_%d', x), 1:results.n_agents, 'UniformOutput', false);
            detection_table = array2table(results.detection_rates', 'VariableNames', agent_names);
            detection_table.Iteration = (1:results.n_iterations)';
            detection_table = detection_table(:, ['Iteration', agent_names]);
            
            writetable(detection_table, sprintf('results/detection_rates_%s.csv', timestamp));
            
            % 资源利用率数据
            resource_table = array2table(results.resource_utilization', 'VariableNames', agent_names);
            resource_table.Iteration = (1:results.n_iterations)';
            resource_table = resource_table(:, ['Iteration', agent_names]);
            
            writetable(resource_table, sprintf('results/resource_utilization_%s.csv', timestamp));
            
            % 汇总统计
            summary_data = DataManager.calculateSummaryStats(results);
            summary_table = struct2table(summary_data);
            writetable(summary_table, sprintf('results/summary_stats_%s.csv', timestamp));
            
            fprintf('✓ CSV文件已导出到results目录\n');
        end
        
        function summary = calculateSummaryStats(results)
            % 计算汇总统计信息
            last_iters = max(1, results.n_iterations-99):results.n_iterations;
            
            for i = 1:results.n_agents
                summary.agent(i).final_detection_rate = mean(results.detection_rates(i, last_iters));
                summary.agent(i).final_resource_util = mean(results.resource_utilization(i, last_iters));
                summary.agent(i).final_convergence = mean(results.convergence_metrics(i, last_iters));
                summary.agent(i).max_detection_rate = max(results.detection_rates(i, :));
                summary.agent(i).min_detection_rate = min(results.detection_rates(i, :));
            end
            
            summary.overall_best_detection = max([summary.agent.final_detection_rate]);
            summary.overall_best_resource = max([summary.agent.final_resource_util]);
            summary.total_iterations = results.n_iterations;
        end
        
        function mergeResults(filenames)
            % 合并多次仿真的结果
            if isempty(filenames)
                error('需要提供文件名列表');
            end
            
            merged_data = [];
            
            for i = 1:length(filenames)
                data = DataManager.loadResults(filenames{i});
                if isempty(merged_data)
                    merged_data = data;
                    merged_data.results.detection_rates_all = data.results.detection_rates;
                    merged_data.results.resource_utilization_all = data.results.resource_utilization;
                else
                    % 合并数据
                    merged_data.results.detection_rates_all = cat(3, ...
                        merged_data.results.detection_rates_all, data.results.detection_rates);
                    merged_data.results.resource_utilization_all = cat(3, ...
                        merged_data.results.resource_utilization_all, data.results.resource_utilization);
                end
            end
            
            % 计算平均值
            merged_data.results.detection_rates_mean = mean(merged_data.results.detection_rates_all, 3);
            merged_data.results.detection_rates_std = std(merged_data.results.detection_rates_all, 0, 3);
            
            % 保存合并结果
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            save(sprintf('results/merged_results_%s.mat', timestamp), 'merged_data');
            fprintf('✓ 合并结果已保存\n');
        end
    end
end