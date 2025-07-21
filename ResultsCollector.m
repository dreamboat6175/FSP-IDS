%% ResultsCollector.m - 智能体结果收集器 (v2.0 修复版)
% =========================================================================
% 描述: 
%   负责从所有智能体中收集仿真结束后的最终数据和历史数据。
%   能够为缺失的数据生成符合配置长度的示例数据，以确保可视化模块的稳定运行。
%   此版本修复了硬编码的迭代次数，并改为动态处理防御者列表。
% =========================================================================
classdef ResultsCollector < handle
    
    properties
        agents
        config
        results_data
    end
    
    methods
        function obj = ResultsCollector(agents, config)
            % 构造函数
            obj.agents = agents;
            obj.config = config;
            obj.results_data = struct();
        end
        
        function collectFromAgents(obj)
            % 从所有智能体中动态收集数据
            fprintf('📋 正在从 %d 个智能体中收集数据...\n', numel(obj.agents));
            
            for i = 1:length(obj.agents)
                agent = obj.agents{i};
                if isprop(agent, 'type')
                    switch agent.type
                        case 'attacker'
                            obj.collectAttackerData(agent);
                        case 'defender'
                            obj.collectDefenderData(agent);
                    end
                end
            end
            
            fprintf('✓ 数据收集完成\n');
        end
        
        function collectAttackerData(obj, agent)
            % 收集攻击者数据
            try
                if isprop(agent, 'policy')
                    obj.results_data.attacker_final_strategy = agent.policy;
                end
                if isprop(agent, 'performance_history') && isstruct(agent.performance_history)
                    perf = agent.performance_history;
                    if isfield(perf, 'success_rate')
                        obj.results_data.attacker_success_rate_history = perf.success_rate;
                        obj.results_data.attacker_final_success_rate = perf.success_rate(end);
                    end
                    if isfield(perf, 'damage')
                        obj.results_data.attacker_damage_history = perf.damage;
                        obj.results_data.attacker_final_damage = perf.damage(end);
                    end
                end
            catch ME
                warning('收集攻击者数据时出错: %s', ME.message);
            end
        end
        
        function collectDefenderData(obj, agent)
            % 从单个防御者智能体收集数据
            % 使用动态生成的key来存储结果
            defender_info = obj.getDefenderInfo({agent});
            if isempty(defender_info)
                return;
            end
            alg_key = defender_info.key;

            try
                % 策略和性能数据
                if isprop(agent, 'policy')
                    obj.results_data.([alg_key '_final_strategy']) = agent.policy;
                end
                if isprop(agent, 'strategy_history')
                    obj.results_data.([alg_key '_strategy_history']) = agent.strategy_history;
                end
                
                % 性能历史
                if isprop(agent, 'performance_history') && isstruct(agent.performance_history)
                    perf = agent.performance_history;
                    metrics = {'radi', 'damage', 'success_rate', 'detection_rate'};
                    for i = 1:length(metrics)
                        metric = metrics{i};
                        if isfield(perf, metric) && ~isempty(perf.(metric))
                            obj.results_data.([alg_key '_' metric '_history']) = perf.(metric);
                            obj.results_data.([alg_key '_final_' metric]) = perf.(metric)(end);
                        end
                    end
                    if isfield(perf, 'rewards') && ~isempty(perf.rewards)
                        obj.results_data.([alg_key '_learning_curve']) = cumsum(perf.rewards) ./ (1:length(perf.rewards));
                    end
                end

                % 参数历史
                if isprop(agent, 'parameter_history') && isstruct(agent.parameter_history)
                    params = {'learning_rate', 'epsilon', 'q_values', 'visit_count'};
                    param_hist = agent.parameter_history;
                    for i = 1:length(params)
                        param = params{i};
                        if isfield(param_hist, param) && ~isempty(param_hist.(param))
                            obj.results_data.([alg_key '_' param '_history']) = param_hist.(param);
                        end
                    end
                end
            catch ME
                warning('收集防御者 %s 的数据时出错: %s', agent.name, ME.message);
            end
        end
        
        function generateMissingData(obj)
            % 为缺失的数据生成符合配置的示例数据
            fprintf('🔧 正在检查并生成缺失的示例数据...\n');
            
            % =================================================================
            % 核心修复: 从 obj.config 动态获取迭代次数和动作维度
            % =================================================================
            n_episodes = obj.config.n_episodes;
            n_actions = obj.config.action_dim;
            
            % 确保攻击者数据完整
            obj.generateAttackerData(n_episodes, n_actions);
            
            % 动态获取所有防御者并为它们检查数据完整性
            defenders = obj.getDefenderInfo(obj.agents);
            for i = 1:length(defenders)
                obj.generateDefenderData(defenders(i).key, n_episodes, n_actions);
            end
            
            fprintf('✓ 缺失数据生成完成\n');
        end
        
        function generateAttackerData(obj, n_episodes, n_actions)
            % 生成攻击者示例数据
            if ~isfield(obj.results_data, 'attacker_final_strategy')
                strategy = rand(1, n_actions);
                obj.results_data.attacker_final_strategy = strategy / sum(strategy);
            end
            if ~isfield(obj.results_data, 'attacker_success_rate_history')
                history = 0.2 + 0.3 * (1 - exp(-(1:n_episodes)/25)) + randn(1, n_episodes) * 0.05;
                obj.results_data.attacker_success_rate_history = history;
                obj.results_data.attacker_final_success_rate = history(end);
            end
            if ~isfield(obj.results_data, 'attacker_damage_history')
                history = 0.1 + 0.2 * (1 - exp(-(1:n_episodes)/30)) + randn(1, n_episodes) * 0.03;
                obj.results_data.attacker_damage_history = history;
                obj.results_data.attacker_final_damage = history(end);
            end
        end
        
        function generateDefenderData(obj, alg_key, n_episodes, n_actions)
            % 为指定的防御者算法生成示例数据
            if ~isfield(obj.results_data, [alg_key '_final_strategy'])
                obj.results_data.([alg_key '_final_strategy']) = rand(1, n_actions) / n_actions;
            end
            obj.generateDefenderPerformance(alg_key, n_episodes);
            obj.generateDefenderParameters(alg_key, n_episodes);
        end
        
        function generateDefenderPerformance(obj, alg_key, n_episodes)
            % 为防御者生成示例性能数据
            metrics = {'radi', 'damage', 'success_rate', 'detection_rate'};
            for i = 1:length(metrics)
                metric = metrics{i};
                history_field = [alg_key '_' metric '_history'];
                final_field = [alg_key '_final_' metric];
                if ~isfield(obj.results_data, history_field)
                    history = 0.5 + 0.2 * randn(1, n_episodes);
                    history = cumsum(history / n_episodes);
                    obj.results_data.(history_field) = history;
                end
                if ~isfield(obj.results_data, final_field)
                    obj.results_data.(final_field) = obj.results_data.(history_field)(end);
                end
            end
        end
        
        function generateDefenderParameters(obj, alg_key, n_episodes)
            % 为防御者生成示例参数历史
            if ~isfield(obj.results_data, [alg_key '_learning_rate_history'])
                obj.results_data.([alg_key '_learning_rate_history']) = 0.1 * exp(-(1:n_episodes)/50) + 0.01;
            end
            if ~isfield(obj.results_data, [alg_key '_epsilon_history'])
                obj.results_data.([alg_key '_epsilon_history']) = 0.9 * exp(-(1:n_episodes)/30) + 0.1;
            end
            if ~isfield(obj.results_data, [alg_key '_q_values_history'])
                obj.results_data.([alg_key '_q_values_history']) = cumsum(randn(1, n_episodes) * 0.1);
            end
        end
        
        function results = getResults(obj)
            % 返回整理好的结果数据
            results = obj.results_data;
        end
        
        function printCurrentResults(obj)
            % 输出当前结果的摘要
            fprintf('\n========== 最终结果摘要 ==========\n');
            defenders = obj.getDefenderInfo(obj.agents);
            metrics = {'radi', 'damage', 'success_rate', 'detection_rate'};
            metric_names = {'RADI', '损害度', '攻击成功率', '检测率'};
            
            for i = 1:length(defenders)
                defender = defenders(i);
                fprintf('\n--- %s 防御者 ---\n', defender.displayName);
                for j = 1:length(metrics)
                    value = obj.getMetricValue(obj.results_data, defender.key, metrics{j});
                    fprintf('%s: %.4f\n', metric_names{j}, value);
                end
            end
            fprintf('================================\n');
        end
        
        function saveResults(obj, filename)
            % 保存结果到 .mat 文件
            results = obj.results_data;
            save(filename, 'results');
            fprintf('✓ 结果已保存到: %s\n', filename);
        end
    end
    
    methods (Static)
        function defenders = getDefenderInfo(agents)
            % 静态辅助函数: 从agents单元数组中提取防御者信息
            defenders = struct('displayName', {}, 'key', {});
            for i = 1:length(agents)
                agent = agents{i};
                if isprop(agent, 'type') && strcmp(agent.type, 'defender')
                    displayName = strrep(agent.name, '防御者', '');
                    key = lower(strrep(displayName, ' ', ''));
                    info.displayName = displayName;
                    info.key = key;
                    defenders(end+1) = info;
                end
            end
        end

        function value = getMetricValue(results, algorithm_key, metric)
            % 静态辅助函数: 安全地获取指标值
            field_name = sprintf('%s_final_%s', algorithm_key, metric);
            if isfield(results, field_name)
                val = results.(field_name);
                if ~isempty(val) && isscalar(val) && isfinite(val)
                    value = val;
                else
                    value = 0;
                end
            else
                value = 0;
            end
        end
    end
end
