%% ResultsCollector.m - 结果收集器类 (修复版)
% =========================================================================
% 描述: 负责收集和组织FSP仿真过程中的所有数据，包括智能体性能、
%      策略演化、资源分配等信息，为可视化和分析提供数据支持
% 版本: v1.2 - 添加缺失的 saveAgentModels 和 saveAllResults 方法
% =========================================================================

classdef ResultsCollector < handle
    properties (Access = private)
        agents_list      % 智能体列表 {attacker, defender1, defender2, ...}
        config          % 配置参数
        results_data    % 收集的结果数据
        n_agents        % 智能体总数
        n_defenders     % 防御者数量
        current_iter    % 当前迭代次数
    end
    
    methods
        function obj = ResultsCollector(agents_list, config)
            % 构造函数
            % 输入:
            %   agents_list - 智能体数组 {attacker, defender1, defender2, ...}
            %   config - 配置结构体
            
            obj.agents_list = agents_list;
            obj.config = config;
            obj.n_agents = length(agents_list);
            obj.n_defenders = obj.n_agents - 1; % 减去攻击者
            obj.current_iter = 1;
            
            % 初始化结果数据结构
            obj.initializeResultsStructure();
            
            fprintf('✓ ResultsCollector初始化完成 (%d个智能体)\n', obj.n_agents);
        end
        
        function initializeResultsStructure(obj)
            % 初始化结果数据结构
            obj.results_data = struct();
            
            % 基本信息
            obj.results_data.n_agents = obj.n_agents;
            obj.results_data.n_defenders = obj.n_defenders;
            % 修复：正确访问嵌套的 n_iterations
            obj.results_data.n_iterations = obj.config.simulation.n_iterations;
            obj.results_data.timestamp = datestr(now);
            
            % 攻击者数据
            obj.results_data.attacker = struct();
            obj.results_data.attacker.name = 'Attacker';
            obj.results_data.attacker.algorithm = 'adaptive';
            obj.results_data.attacker.performance = struct();
            
            % 防御者数据
            obj.results_data.defenders = struct();
            
            % 为每个防御者初始化数据结构
            for i = 1:obj.n_defenders
                defender_name = sprintf('defender%d', i);
                obj.results_data.defenders.(defender_name) = struct();
                obj.results_data.defenders.(defender_name).name = defender_name;
                obj.results_data.defenders.(defender_name).algorithm = 'unknown';
                obj.results_data.defenders.(defender_name).performance = struct();
            end
            
            % 全局统计数据
            obj.results_data.global_stats = struct();
            obj.results_data.global_stats.total_episodes = 0;
            obj.results_data.global_stats.total_steps = 0;
            obj.results_data.global_stats.convergence_achieved = false;
        end
        
        function collectFromAgents(obj)
            % 从智能体收集性能数据
            fprintf('📊 从智能体收集性能数据...\n');
            
            try
                % 收集攻击者数据
                if ~isempty(obj.agents_list{1})
                    obj.collectAttackerData(obj.agents_list{1});
                end
                
                % 收集防御者数据
                for i = 2:obj.n_agents
                    if ~isempty(obj.agents_list{i})
                        obj.collectDefenderData(obj.agents_list{i}, i-1);
                    end
                end
                
                fprintf('✓ 智能体数据收集完成\n');
                
            catch ME
                fprintf('⚠️ 数据收集过程中出现错误: %s\n', ME.message);
                % 继续执行，不中断程序
            end
        end
        
        function collectAttackerData(obj, attacker)
            % 收集攻击者数据
            try
                % 基本信息
                if isprop(attacker, 'name') || isfield(attacker, 'name')
                    obj.results_data.attacker.name = attacker.name;
                end
                
                if isprop(attacker, 'algorithm') || isfield(attacker, 'algorithm')
                    obj.results_data.attacker.algorithm = attacker.algorithm;
                end
                
                % 性能指标
                performance = struct();
                
                % 尝试获取各种性能指标
                if ismethod(attacker, 'getPerformanceMetrics')
                    metrics = attacker.getPerformanceMetrics();
                    performance = metrics;
                elseif isprop(attacker, 'performance_history') || isfield(attacker, 'performance_history')
                    performance.history = attacker.performance_history;
                end
                
                % 获取累积奖励
                if isprop(attacker, 'total_reward') || isfield(attacker, 'total_reward')
                    performance.total_reward = attacker.total_reward;
                end
                
                obj.results_data.attacker.performance = performance;
                
            catch ME
                fprintf('⚠️ 攻击者数据收集失败: %s\n', ME.message);
            end
        end
        
        function collectDefenderData(obj, defender, defender_idx)
            % 收集防御者数据
            try
                defender_name = sprintf('defender%d', defender_idx);
                
                % 基本信息
                if isprop(defender, 'name') || isfield(defender, 'name')
                    obj.results_data.defenders.(defender_name).name = defender.name;
                else
                    obj.results_data.defenders.(defender_name).name = defender_name;
                end
                
                if isprop(defender, 'algorithm') || isfield(defender, 'algorithm')
                    obj.results_data.defenders.(defender_name).algorithm = defender.algorithm;
                end
                
                % 性能指标
                performance = struct();
                
                % 尝试获取各种性能指标
                if ismethod(defender, 'getPerformanceMetrics')
                    metrics = defender.getPerformanceMetrics();
                    performance = metrics;
                elseif isprop(defender, 'performance_history') || isfield(defender, 'performance_history')
                    performance.history = defender.performance_history;
                end
                
                obj.results_data.defenders.(defender_name).performance = performance;
                
            catch ME
                fprintf('⚠️ 防御者%d数据收集失败: %s\n', defender_idx, ME.message);
            end
        end
        
        function printSummary(obj)
            % 打印性能摘要
            fprintf('\n========================\n');
            fprintf('🎯 FSP仿真性能摘要\n');
            fprintf('========================\n');
            
            % 攻击者摘要
            fprintf('攻击者 (%s):\n', obj.results_data.attacker.algorithm);
            if isfield(obj.results_data.attacker, 'performance')
                perf = obj.results_data.attacker.performance;
                if isfield(perf, 'total_reward') && ~isempty(perf.total_reward)
                    fprintf('  总奖励: %.2f\n', perf.total_reward);
                end
            end
            
            % 防御者摘要
            defender_names = fieldnames(obj.results_data.defenders);
            for i = 1:length(defender_names)
                defender_name = defender_names{i};
                defender = obj.results_data.defenders.(defender_name);
                
                fprintf('  %s (%s):\n', defender.name, defender.algorithm);
                
                if isfield(defender, 'performance')
                    perf = defender.performance;
                    if isfield(perf, 'radi') && ~isempty(perf.radi)
                        fprintf('    RADI: %.4f\n', perf.radi(end));
                    end
                    if isfield(perf, 'efficiency') && ~isempty(perf.efficiency)
                        fprintf('    效率: %.2f%%\n', perf.efficiency(end) * 100);
                    end
                    if isfield(perf, 'detection_rate') && ~isempty(perf.detection_rate)
                        fprintf('    检测率: %.2f%%\n', perf.detection_rate(end) * 100);
                    end
                end
            end
            
            fprintf('========================\n\n');
        end
        
        function results = getResults(obj)
            % 获取收集的结果数据
            results = obj.results_data;r
        end
        
        function updateIterationData(obj, iteration, episode_results)
            % 更新迭代数据（为兼容性保留）
            obj.current_iter = iteration;
            % 这里可以添加具体的更新逻辑
        end

        function saveCheckpoint(obj, iteration, attacker_agent, defender_agents)
            % saveCheckpoint - 保存当前仿真状态作为检查点
            % Inputs:
            %   iteration - 当前迭代次数
            %   attacker_agent - 攻击者智能体对象
            %   defender_agents - 防御者智能体对象（cell 数组）
        
            try
                % 1. 确保 results 目录存在
                output_dir = fullfile(pwd, 'results'); % 使用完整路径
                if ~exist(output_dir, 'dir')
                    [success, msg] = mkdir(output_dir);
                    if ~success
                        error('无法创建results目录: %s', msg);
                    end
                    fprintf('📁 创建目录: %s\n', output_dir);
                end
        
                % 2. 生成时间戳和文件名
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                checkpoint_filename = fullfile(output_dir, sprintf('checkpoint_iter_%d_%s.mat', iteration, timestamp));
                
                % 3. 验证输入参数
                if isempty(attacker_agent)
                    warning('攻击者智能体为空，使用默认值');
                    attacker_agent = struct('name', 'AttackerAgent', 'type', 'default');
                end
                
                if isempty(defender_agents)
                    warning('防御者智能体为空，使用默认值');
                    defender_agents = {struct('name', 'DefenderAgent', 'type', 'default')};
                end
                
                % 4. 确保 results_data 存在
                if isempty(obj.results_data)
                    obj.results_data = struct();
                    obj.results_data.timestamp = timestamp;
                    obj.results_data.status = 'checkpoint_created';
                    warning('results_data 为空，已创建默认结构');
                end
                
                % 5. 创建检查点数据结构
                checkpoint_data = struct();
                checkpoint_data.iteration = iteration;
                checkpoint_data.timestamp = timestamp;
                checkpoint_data.matlab_version = version;
                
                % 安全保存智能体数据
                try
                    checkpoint_data.attacker_agent_model = attacker_agent;
                catch ME
                    warning('攻击者智能体保存失败: %s', ME.message);
                    checkpoint_data.attacker_agent_model = struct('error', ME.message);
                end
                
                try
                    checkpoint_data.defender_agent_models = defender_agents;
                catch ME
                    warning('防御者智能体保存失败: %s', ME.message);
                    checkpoint_data.defender_agent_models = {struct('error', ME.message)};
                end
                
                try
                    checkpoint_data.current_results_data = obj.results_data;
                catch ME
                    warning('结果数据保存失败: %s', ME.message);
                    checkpoint_data.current_results_data = struct('error', ME.message);
                end
                
                % 6. 保存检查点文件
                save(checkpoint_filename, 'checkpoint_data', '-v7.3');
                fprintf('✓ 检查点已保存: %s\n', checkpoint_filename);
                
                % 7. 记录到日志
                if exist('Logger', 'class') == 8
                    Logger.info(sprintf('检查点已保存: %s', checkpoint_filename));
                end
                
            catch ME
                error_msg = sprintf('保存检查点失败: %s\n位置: %s (第%d行)', ...
                                   ME.message, ME.stack(1).file, ME.stack(1).line);
                
                if exist('Logger', 'class') == 8
                    Logger.error(error_msg);
                end
                
                fprintf('❌ %s\n', error_msg);
                
                % 创建简化的检查点作为备用
                try
                    backup_filename = fullfile(pwd, sprintf('backup_checkpoint_iter_%d.mat', iteration));
                    backup_data = struct('iteration', iteration, 'timestamp', datestr(now), 'error', ME.message);
                    save(backup_filename, 'backup_data');
                    fprintf('📋 备用检查点已保存: %s\n', backup_filename);
                catch
                    fprintf('❌ 连备用检查点也无法保存\n');
                end
            end
        end        

        
        function saveAgentModels(obj, attacker_agent, defender_agents)
            % saveAgentModels - 保存最终训练好的智能体模型
            % Inputs:
            %   attacker_agent - 攻击者智能体对象
            %   defender_agents - 防御者智能体对象（cell 数组）
            
            try
                output_dir = fullfile('results');
                if ~exist(output_dir, 'dir')
                    mkdir(output_dir);
                end
                
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                models_filename = fullfile(output_dir, sprintf('trained_models_%s.mat', timestamp));
                
                % 创建模型保存结构
                models_data = struct();
                models_data.timestamp = timestamp;
                models_data.training_config = obj.config;
                
                % 保存攻击者模型
                try
                    if ismethod(attacker_agent, 'saveModel')
                        models_data.attacker_model = attacker_agent.saveModel();
                    else
                        models_data.attacker_model = attacker_agent;
                    end
                    models_data.attacker_info.name = 'Attacker';
                    if isprop(attacker_agent, 'algorithm') || isfield(attacker_agent, 'algorithm')
                        models_data.attacker_info.algorithm = attacker_agent.algorithm;
                    end
                catch ME_att
                    fprintf('⚠️ 攻击者模型保存失败: %s\n', ME_att.message);
                    models_data.attacker_model = [];
                end
                
                % 保存防御者模型
                models_data.defender_models = cell(length(defender_agents), 1);
                models_data.defender_info = cell(length(defender_agents), 1);
                
                for i = 1:length(defender_agents)
                    try
                        if ismethod(defender_agents{i}, 'saveModel')
                            models_data.defender_models{i} = defender_agents{i}.saveModel();
                        else
                            models_data.defender_models{i} = defender_agents{i};
                        end
                        
                        % 保存防御者信息
                        defender_info = struct();
                        defender_info.index = i;
                        if isprop(defender_agents{i}, 'name') || isfield(defender_agents{i}, 'name')
                            defender_info.name = defender_agents{i}.name;
                        else
                            defender_info.name = sprintf('Defender_%d', i);
                        end
                        if isprop(defender_agents{i}, 'algorithm') || isfield(defender_agents{i}, 'algorithm')
                            defender_info.algorithm = defender_agents{i}.algorithm;
                        end
                        models_data.defender_info{i} = defender_info;
                        
                    catch ME_def
                        fprintf('⚠️ 防御者%d模型保存失败: %s\n', i, ME_def.message);
                        models_data.defender_models{i} = [];
                        models_data.defender_info{i} = struct('index', i, 'name', sprintf('Defender_%d', i), 'error', ME_def.message);
                    end
                end
                
                % 保存到文件
                save(models_filename, '-struct', 'models_data', '-v7.3');
                fprintf('✓ 智能体模型已保存: %s\n', models_filename);
                Logger.info(sprintf('智能体模型已保存: %s', models_filename));
                
            catch ME
                Logger.error(sprintf('智能体模型保存失败: %s', ME.message));
                fprintf('❌ 智能体模型保存失败: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function saveAllResults(obj)
            % saveAllResults - 保存所有收集到的结果数据
            
            try
                output_dir = fullfile('results');
                if ~exist(output_dir, 'dir')
                    mkdir(output_dir);
                end
                
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                results_filename = fullfile(output_dir, sprintf('simulation_results_%s.mat', timestamp));
                
                % 确保从智能体收集最新数据
                obj.collectFromAgents();
                
                % 创建完整的结果保存结构
                save_data = struct();
                save_data.results_data = obj.results_data;
                save_data.config = obj.config;
                save_data.timestamp = timestamp;
                save_data.matlab_version = version;
                save_data.system_info = struct();
                
                % 添加系统信息
                try
                    save_data.system_info.computer = computer;
                    save_data.system_info.username = getenv('USERNAME');
                    save_data.system_info.save_time = now;
                catch
                    save_data.system_info.note = '系统信息收集失败';
                end
                
                % 计算汇总统计
                try
                    save_data.summary_stats = obj.calculateSummaryStatistics();
                catch ME_summary
                    fprintf('⚠️ 汇总统计计算失败: %s\n', ME_summary.message);
                    save_data.summary_stats = struct();
                end
                
                % 保存到文件
                save(results_filename, '-struct', 'save_data', '-v7.3');
                fprintf('✓ 仿真结果已保存: %s\n', results_filename);
                Logger.info(sprintf('仿真结果已保存: %s', results_filename));
                
                % 同时保存为CSV格式（如果可能）
                try
                    obj.exportToCSV(timestamp);
                catch ME_csv
                    fprintf('⚠️ CSV导出失败: %s\n', ME_csv.message);
                end
                
            catch ME
                Logger.error(sprintf('结果保存失败: %s', ME.message));
                fprintf('❌ 结果保存失败: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function summary_stats = calculateSummaryStatistics(obj)
            % 计算汇总统计信息
            summary_stats = struct();
            
            try
                % 基本统计
                summary_stats.n_agents = obj.results_data.n_agents;
                summary_stats.n_defenders = obj.results_data.n_defenders;
                summary_stats.n_iterations = obj.results_data.n_iterations;
                
                % 攻击者统计
                if isfield(obj.results_data.attacker, 'performance')
                    perf = obj.results_data.attacker.performance;
                    if isfield(perf, 'total_reward')
                        summary_stats.attacker_total_reward = perf.total_reward;
                    end
                end
                
                % 防御者统计
                defender_names = fieldnames(obj.results_data.defenders);
                summary_stats.defender_performance = struct();
                
                for i = 1:length(defender_names)
                    defender_name = defender_names{i};
                    if isfield(obj.results_data.defenders.(defender_name), 'performance')
                        summary_stats.defender_performance.(defender_name) = ...
                            obj.results_data.defenders.(defender_name).performance;
                    end
                end
                
                % 时间统计
                summary_stats.simulation_start = obj.results_data.timestamp;
                summary_stats.simulation_end = datestr(now);
                
            catch ME
                fprintf('⚠️ 汇总统计计算部分失败: %s\n', ME.message);
                summary_stats.error = ME.message;
            end
        end
        
        function exportToCSV(obj, timestamp)
            % 导出结果到CSV文件
            try
                output_dir = fullfile('results');
                csv_filename = fullfile(output_dir, sprintf('summary_results_%s.csv', timestamp));
                
                % 创建汇总表格
                summary_table = table();
                
                % 添加攻击者数据
                if isfield(obj.results_data.attacker, 'performance')
                    attacker_row = table();
                    attacker_row.Agent = {'Attacker'};
                    attacker_row.Type = {'Attacker'};
                    attacker_row.Algorithm = {obj.results_data.attacker.algorithm};
                    
                    if isfield(obj.results_data.attacker.performance, 'total_reward')
                        attacker_row.TotalReward = obj.results_data.attacker.performance.total_reward;
                    else
                        attacker_row.TotalReward = NaN;
                    end
                    
                    summary_table = [summary_table; attacker_row];
                end
                
                % 添加防御者数据
                defender_names = fieldnames(obj.results_data.defenders);
                for i = 1:length(defender_names)
                    defender_name = defender_names{i};
                    defender = obj.results_data.defenders.(defender_name);
                    
                    defender_row = table();
                    defender_row.Agent = {defender.name};
                    defender_row.Type = {'Defender'};
                    defender_row.Algorithm = {defender.algorithm};
                    defender_row.TotalReward = NaN; % 防御者通常没有total_reward
                    
                    summary_table = [summary_table; defender_row];
                end
                
                % 写入CSV文件
                if ~isempty(summary_table)
                    writetable(summary_table, csv_filename);
                    fprintf('✓ CSV摘要已导出: %s\n', csv_filename);
                end
                
            catch ME
                fprintf('⚠️ CSV导出失败: %s\n', ME.message);
            end
        end
    end
end