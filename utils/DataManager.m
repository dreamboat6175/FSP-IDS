%% DataManager.m - 数据管理器类 (修复版)
% =========================================================================
% 描述: 管理仿真结果的保存、加载和导出功能
% 修复版本: 解决表格行数不匹配的错误
% =========================================================================

classdef DataManager
    methods (Static)
        function saveResults(results, config, agents)
            % 保存仿真结果
            
            % 如果没有提供agents参数，设为空结构
            if nargin < 3
                agents = struct();
            end
            
            fprintf('💾 开始保存仿真结果...\n');
            
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            if ~exist('results', 'dir')
                mkdir('results');
            end
            
            % 创建保存数据结构
            save_data = struct();
            save_data.results = results;
            save_data.config = config;
            save_data.timestamp = now;
            save_data.matlab_version = version;
            save_data.policies = {};
            save_data.agent_names = {};
            
            % 安全地保存智能体数据
            try
                if isfield(agents, 'defenders') && ~isempty(agents.defenders)
                    for i = 1:length(agents.defenders)
                        if ismethod(agents.defenders{i}, 'getPolicy')
                            save_data.policies{i} = agents.defenders{i}.getPolicy();
                        end
                        if isprop(agents.defenders{i}, 'name')
                            save_data.agent_names{i} = agents.defenders{i}.name;
                        else
                            save_data.agent_names{i} = sprintf('Defender_%d', i);
                        end
                        if ismethod(agents.defenders{i}, 'getStatistics')
                            save_data.agent_stats{i} = agents.defenders{i}.getStatistics();
                        end
                    end
                end
                
                if isfield(agents, 'attacker') && ~isempty(agents.attacker)
                    if ismethod(agents.attacker, 'getPolicy')
                        save_data.attacker_policy = agents.attacker.getPolicy();
                    end
                    if ismethod(agents.attacker, 'getStatistics')
                        save_data.attacker_stats = agents.attacker.getStatistics();
                    end
                end
            catch ME
                fprintf('警告: 智能体数据保存失败: %s\n', ME.message);
            end
            
            % 计算汇总统计
            try
                save_data.summary = DataManager.calculateSummaryStats(results);
            catch ME
                fprintf('警告: 汇总统计计算失败: %s\n', ME.message);
                save_data.summary = struct();
            end
            
            % 保存MAT文件
            filename = sprintf('results/fsp_results_%s.mat', timestamp);
            try
                save(filename, 'save_data', '-v7.3');
                fprintf('✓ MAT文件已保存: %s\n', filename);
            catch ME
                fprintf('❌ MAT文件保存失败: %s\n', ME.message);
            end
            
            % 导出CSV文件
            try
                DataManager.exportToCSV(results, timestamp);
                fprintf('✓ CSV文件导出完成\n');
            catch ME
                fprintf('❌ CSV导出失败: %s\n', ME.message);
                fprintf('错误详情: %s\n', ME.getReport());
            end
        end
        
        function data = loadResults(filename)
            % 加载仿真结果
            if exist(filename, 'file')
                try
                    loaded = load(filename);
                    data = loaded.save_data;
                    fprintf('✓ 结果已加载: %s\n', filename);
                catch ME
                    error('加载文件失败: %s - %s', filename, ME.message);
                end
            else
                error('文件不存在: %s', filename);
            end
        end
        
        function exportToCSV(results, timestamp)
            % 导出关键数据为CSV格式 (修复版)
            
            if ~exist('results', 'dir')
                mkdir('results');
            end
            
            fprintf('📊 开始导出CSV文件...\n');
            
            % 安全获取迭代次数和智能体数量
            [n_iterations, n_agents] = DataManager.getDataDimensions(results);
            fprintf('  数据维度: %d 迭代, %d 智能体\n', n_iterations, n_agents);
            
            % 生成智能体名称
            agent_names = arrayfun(@(x) sprintf('Agent_%d', x), 1:n_agents, 'UniformOutput', false);
            
            % 导出各种数据类型
            DataManager.exportSingleMetric(results, 'radi', agent_names, n_iterations, timestamp);
            DataManager.exportSingleMetric(results, 'resource_efficiency', agent_names, n_iterations, timestamp);
            DataManager.exportSingleMetric(results, 'allocation_balance', agent_names, n_iterations, timestamp);
            DataManager.exportSingleMetric(results, 'convergence_metrics', agent_names, n_iterations, timestamp);
            DataManager.exportSingleMetric(results, 'defender_rewards', agent_names, n_iterations, timestamp);
            DataManager.exportSingleMetric(results, 'success_rates', agent_names, n_iterations, timestamp);
            
            % 导出汇总统计
            try
                summary_data = DataManager.calculateSummaryStats(results);
                if ~isempty(fieldnames(summary_data))
                    summary_table = struct2table(summary_data);
                    summary_filename = sprintf('results/summary_stats_%s.csv', timestamp);
                    writetable(summary_table, summary_filename);
                    fprintf('  ✓ 汇总统计: %s\n', summary_filename);
                end
            catch ME
                fprintf('  ❌ 汇总统计导出失败: %s\n', ME.message);
            end
            
            fprintf('✓ CSV文件导出完成\n');
        end
        
        function exportSingleMetric(results, metric_name, agent_names, n_iterations, timestamp)
            % 导出单个指标到CSV文件
            try
                if ~isfield(results, metric_name)
                    return;  % 如果字段不存在，跳过
                end
                
                data_matrix = results.(metric_name);
                if isempty(data_matrix)
                    return;  % 如果数据为空，跳过
                end
                
                % 确保数据维度正确
                [rows, cols] = size(data_matrix);
                
                % 处理数据维度
                if rows == length(agent_names) && cols >= n_iterations
                    % 数据格式: [n_agents, n_iterations]
                    data_to_export = data_matrix(:, 1:n_iterations)';  % 转置为 [n_iterations, n_agents]
                elseif cols == length(agent_names) && rows >= n_iterations
                    % 数据已经是 [n_iterations, n_agents] 格式
                    data_to_export = data_matrix(1:n_iterations, :);
                else
                    % 数据维度不匹配，尝试调整
                    if rows == n_iterations
                        % 取前n_agents列
                        data_to_export = data_matrix(:, 1:min(cols, length(agent_names)));
                        % 如果列数不足，用零填充
                        if size(data_to_export, 2) < length(agent_names)
                            pad_cols = length(agent_names) - size(data_to_export, 2);
                            data_to_export = [data_to_export, zeros(n_iterations, pad_cols)];
                        end
                        agent_names_used = agent_names(1:size(data_to_export, 2));
                    else
                        fprintf('  ⚠️ %s 数据维度不匹配 [%d×%d]，跳过\n', metric_name, rows, cols);
                        return;
                    end
                end
                
                % 确保使用正确的智能体名称
                if ~exist('agent_names_used', 'var')
                    agent_names_used = agent_names(1:size(data_to_export, 2));
                end
                
                % 创建表格
                data_table = array2table(data_to_export, 'VariableNames', agent_names_used);
                
                % 添加迭代列 - 确保长度匹配
                iteration_column = (1:size(data_to_export, 1))';
                data_table.Iteration = iteration_column;
                
                % 重新排列列顺序（Iteration在前）
                data_table = data_table(:, ['Iteration', agent_names_used]);
                
                % 保存文件
                filename = sprintf('results/%s_%s.csv', metric_name, timestamp);
                writetable(data_table, filename);
                fprintf('  ✓ %s: %s\n', metric_name, filename);
                
            catch ME
                fprintf('  ❌ %s 导出失败: %s\n', metric_name, ME.message);
            end
        end
        
        function [n_iterations, n_agents] = getDataDimensions(results)
            % 安全获取数据维度
            n_iterations = 100;  % 默认值
            n_agents = 3;        % 默认值
            
            % 尝试从不同字段获取迭代次数
            if isfield(results, 'n_iterations') && ~isempty(results.n_iterations)
                n_iterations = results.n_iterations;
            elseif isfield(results, 'defender_rewards') && ~isempty(results.defender_rewards)
                n_iterations = max(size(results.defender_rewards));
            elseif isfield(results, 'radi') && ~isempty(results.radi)
                n_iterations = max(size(results.radi));
            elseif isfield(results, 'attacker_rewards') && ~isempty(results.attacker_rewards)
                n_iterations = length(results.attacker_rewards);
            end
            
            % 尝试从不同字段获取智能体数量
            if isfield(results, 'n_agents') && ~isempty(results.n_agents)
                n_agents = results.n_agents;
            elseif isfield(results, 'defender_rewards') && ~isempty(results.defender_rewards)
                n_agents = min(size(results.defender_rewards));
            elseif isfield(results, 'radi') && ~isempty(results.radi)
                n_agents = min(size(results.radi));
            end
            
            % 确保维度合理
            n_iterations = max(1, n_iterations);
            n_agents = max(1, n_agents);
        end
        
        function summary = calculateSummaryStats(results)
            % 计算汇总统计 (修复版)
            summary = struct();
            
            try
                % 获取数据维度
                [n_iterations, n_agents] = DataManager.getDataDimensions(results);
                summary.total_iterations = n_iterations;
                summary.total_agents = n_agents;
                
                % 计算最后几次迭代的平均值
                last_iters = max(1, n_iterations-9):n_iterations;  % 最后10次迭代
                
                % 初始化字段
                summary.overall_best_radi = NaN;
                summary.overall_best_efficiency = NaN;
                summary.overall_best_balance = NaN;
                summary.final_performance = struct();
                
                % 处理各种指标
                metrics_to_process = {'radi', 'resource_efficiency', 'allocation_balance', ...
                                    'defender_rewards', 'success_rates', 'convergence_metrics'};
                
                for i = 1:length(metrics_to_process)
                    metric = metrics_to_process{i};
                    if isfield(results, metric) && ~isempty(results.(metric))
                        data_matrix = results.(metric);
                        
                        % 确保数据格式正确
                        [rows, cols] = size(data_matrix);
                        if rows == n_agents && cols >= length(last_iters)
                            % 数据格式: [n_agents, n_iterations]
                            recent_data = data_matrix(:, last_iters);
                        elseif cols == n_agents && rows >= length(last_iters)
                            % 数据格式: [n_iterations, n_agents]
                            recent_data = data_matrix(last_iters, :)';
                        else
                            continue;  % 跳过维度不匹配的数据
                        end
                        
                        % 计算统计值
                        mean_values = mean(recent_data, 2);
                        final_values = recent_data(:, end);
                        
                        summary.final_performance.(metric) = struct();
                        summary.final_performance.(metric).mean_recent = mean(mean_values);
                        summary.final_performance.(metric).std_recent = std(mean_values);
                        summary.final_performance.(metric).final_values = final_values;
                        
                        % 更新总体最优值
                        switch metric
                            case 'radi'
                                summary.overall_best_radi = min(final_values);
                            case 'resource_efficiency'
                                summary.overall_best_efficiency = max(final_values);
                            case 'allocation_balance'
                                summary.overall_best_balance = max(final_values);
                        end
                    end
                end
                
                % 添加时间戳
                summary.calculation_time = datestr(now, 'yyyy-mm-dd HH:MM:SS');
                
            catch ME
                fprintf('汇总统计计算出错: %s\n', ME.message);
                % 返回基本结构
                summary.total_iterations = 100;
                summary.total_agents = 3;
                summary.calculation_error = ME.message;
            end
        end
        
        function mergeResults(filenames)
            % 合并多次仿真的结果
            if isempty(filenames)
                error('需要提供文件名列表');
            end
            
            fprintf('🔗 开始合并 %d 个结果文件...\n', length(filenames));
            
            merged_data = [];
            valid_files = 0;
            
            for i = 1:length(filenames)
                try
                    data = DataManager.loadResults(filenames{i});
                    valid_files = valid_files + 1;
                    
                    if isempty(merged_data)
                        merged_data = data;
                        % 初始化合并字段
                        metrics = {'radi', 'resource_efficiency', 'allocation_balance'};
                        for j = 1:length(metrics)
                            if isfield(data.results, metrics{j})
                                merged_data.results.([metrics{j} '_all']) = data.results.(metrics{j});
                            end
                        end
                    else
                        % 合并数据
                        for j = 1:length(metrics)
                            if isfield(data.results, metrics{j}) && isfield(merged_data.results, [metrics{j} '_all'])
                                merged_data.results.([metrics{j} '_all']) = cat(3, ...
                                    merged_data.results.([metrics{j} '_all']), data.results.(metrics{j}));
                            end
                        end
                    end
                    
                    fprintf('  ✓ 文件 %d/%d: %s\n', i, length(filenames), filenames{i});
                    
                catch ME
                    fprintf('  ❌ 文件 %d/%d 加载失败: %s\n', i, length(filenames), ME.message);
                end
            end
            
            if valid_files > 0
                % 计算平均值和标准差
                metrics = {'radi', 'resource_efficiency', 'allocation_balance'};
                for i = 1:length(metrics)
                    all_field = [metrics{i} '_all'];
                    if isfield(merged_data.results, all_field)
                        merged_data.results.([metrics{i} '_mean']) = mean(merged_data.results.(all_field), 3);
                        merged_data.results.([metrics{i} '_std']) = std(merged_data.results.(all_field), 0, 3);
                    end
                end
                
                % 保存合并结果
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                save_filename = sprintf('results/merged_results_%s.mat', timestamp);
                save(save_filename, 'merged_data', '-v7.3');
                fprintf('✓ 合并结果已保存: %s\n', save_filename);
                fprintf('✓ 成功合并 %d 个文件\n', valid_files);
            else
                error('没有有效的文件可以合并');
            end
        end
    end
end