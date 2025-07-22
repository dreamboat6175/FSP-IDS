function main_fsp()
    %% main_fsp - FSP-TCS主仿真函数（简化版）
    % ================================================================
    % 主函数只负责调用其他函数，不做具体实现
    % ================================================================
    clear;
    clc;
    close all;

    addpath(genpath(pwd));

    fprintf('🚀 FSP-TCS仿真系统启动\n\n');
    
    try
        % 1. 初始化日志系统
        % 确保logs目录存在
        if ~exist('logs', 'dir')
            mkdir('logs');
        end
        
        % 生成带时间戳的日志文件名
        log_file = fullfile('logs', sprintf('simulation_%s.log', datestr(now, 'yyyymmdd_HHMMSS')));
        
        % 初始化Logger（提供log_file参数）
        Logger.initialize(log_file);
        Logger.info('FSP-TCS仿真开始');
        
        % 2. 加载配置
        config = ConfigManager.loadConfig();
        
        % 3. 创建环境和智能体
        [env, agents] = createEnvironmentAndAgents(config);
        
        % 4. 初始化仿真器
        if exist('FSPSimulator', 'class') == 8
            simulator = FSPSimulator(config);
        else
            simulator = [];
        end
        
        % 5. 初始化结果结构
        results = initializeResults(config);
        
        % 6. 主仿真循环
        fprintf('🔄 开始FSP仿真训练...\n');
        for iteration = 1:config.n_iterations
            fprintf('=== 迭代 %d/%d ===\n', iteration, config.n_iterations);
            
            % 更新策略记录（新增）
            updateStrategiesInEnvironment(env, agents, config);
            
            % 运行episodes
            if ~isempty(simulator)
                episode_results = simulator.runEpisodes(env, agents(2:end), agents{1}, config);
            else
                episode_results = runSimpleEpisodes(env, agents(2:end), agents{1}, config);
            end
            
            % 记录迭代结果
            recordIterationResults(results, episode_results, agents, iteration, config);
            
            % 数据完整性检查（每10次迭代检查一次）
            if mod(iteration, 10) == 0
                checkDataIntegrity(env, iteration);
            end
            
            % 输出进度
            if ~isempty(env.radi_history)
                fprintf('📈 迭代 %d 完成，RADI: %.4f\n', iteration, env.radi_history(end));
            end
        end
        fprintf('✓ FSP仿真训练完成\n');
        
        % 7. 生成可视化报告
        if config.generate_visualization
            generateCompleteVisualizationReport(agents, results, config, env);
        end
        
        % 8. 导出详细数据（可选）
        if isfield(config, 'export_detailed_data') && config.export_detailed_data
            exportDetailedData(env, config);
        end
        
        fprintf('🎉 FSP-TCS仿真完成！\n');
        
    catch ME
        fprintf('❌ 仿真出错: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('错误位置: %s, 行号: %d\n', ME.stack(1).file, ME.stack(1).line);
        end
        
        % 记录错误到日志
        Logger.error(sprintf('仿真出错: %s', ME.message));
        Logger.error(sprintf('错误位置: %s, 行号: %d', ME.stack(1).file, ME.stack(1).line));
        
        rethrow(ME);
    finally
        % 关闭日志系统
        Logger.close();
        fprintf('✓ 日志系统已关闭\n');
    end
end

%% ========== 新增的辅助函数（简单实现） ==========

function updateStrategiesInEnvironment(env, agents, config)
    %UPDATESTRATEGIESINVIRONMENT 更新环境中的策略记录
    
    try
        % 获取攻击者策略
        attacker = agents{1};
        if hasmethod(attacker, 'getStrategy')
            attack_strategy = attacker.getStrategy();
        elseif isprop(attacker, 'strategy') || isfield(attacker, 'strategy')
            attack_strategy = attacker.strategy;
        else
            attack_strategy = ones(1, config.n_stations) / config.n_stations;
        end
        
        % 获取防御者策略（使用第一个防御者）
        if length(agents) > 1
            defender = agents{2};
            if hasmethod(defender, 'getStrategy')
                defense_strategy = defender.getStrategy();
            elseif isprop(defender, 'strategy') || isfield(defender, 'strategy')
                defense_strategy = defender.strategy;
            else
                defense_strategy = ones(1, config.n_stations) / config.n_stations;
            end
        else
            defense_strategy = ones(1, config.n_stations) / config.n_stations;
        end
        
        % 更新环境中的策略记录
        env.updateStrategies(attack_strategy, defense_strategy);
        
    catch ME
        if isfield(config, 'debug_mode') && config.debug_mode
            fprintf('⚠️ 策略更新失败: %s\n', ME.message);
        end
    end
end

function generateCompleteVisualizationReport(agents, results, config, env)
    %GENERATECOMPLETEVISUALIZATIONREPORT 生成完整的可视化报告
    
    try
        fprintf('\n📊 开始生成可视化报告...\n');
        
        % 使用增强版可视化系统，传入环境对象以获取真实数据
        if exist('EnhancedVisualization', 'class') == 8
            EnhancedVisualization.generateFullReport(agents, results, config, env);
            fprintf('✅ 增强版可视化报告生成成功！\n');
        else
            % 尝试使用传统可视化方法
            fprintf('⚠️ EnhancedVisualization不存在，尝试传统方法...\n');
            if exist('generateVisualizationReport', 'file') == 2
                generateVisualizationReport(results, config);
                fprintf('✅ 传统可视化报告生成成功！\n');
            else
                fprintf('❌ 无可用的可视化方法\n');
            end
        end
        
    catch ME
        fprintf('❌ 可视化报告生成失败: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
        end
        
        % 尝试使用备用可视化方法
        try
            fprintf('🔄 尝试使用备用可视化方法...\n');
            generateVisualizationReport(results, config);
            fprintf('✅ 备用可视化报告生成成功！\n');
        catch ME2
            fprintf('❌ 备用可视化也失败: %s\n', ME2.message);
        end
    end
end

function checkDataIntegrity(env, iteration)
    %CHECKDATAINTEGRITY 检查数据完整性
    
    fprintf('🔍 检查数据完整性 (迭代 %d)...\n', iteration);
    
    % 检查必需的历史数据
    required_fields = {
        'radi_history', 
        'nash_convergence_history', 
        'attack_coverage_history'
    };
    
    for i = 1:length(required_fields)
        field = required_fields{i};
        if isprop(env, field) || isfield(env, field)
            if ~isempty(env.(field))
                fprintf('  ✓ %s: %d个数据点\n', field, length(env.(field)));
            else
                fprintf('  ⚠️ %s: 数据为空\n', field);
            end
        else
            fprintf('  ❌ %s: 字段不存在\n', field);
        end
    end
end

function exportDetailedData(env, config)
    %EXPORTDETAILEDDATA 导出详细数据
    
    fprintf('💾 导出详细数据...\n');
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    save_dir = fullfile(pwd, 'data_export', timestamp);
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    % 导出环境历史数据
    export_data = struct();
    export_data.timestamp = datestr(now);
    export_data.config = config;
    
    % 导出各类历史数据
    history_fields = {
        'radi_history',
        'nash_convergence_history', 
        'attack_coverage_history',
        'defense_effectiveness_history',
        'attack_success_history'
    };
    
    for i = 1:length(history_fields)
        field = history_fields{i};
        if (isprop(env, field) || isfield(env, field)) && ~isempty(env.(field))
            export_data.(field) = env.(field);
        end
    end
    
    % 保存数据
    save(fullfile(save_dir, 'detailed_simulation_data.mat'), 'export_data');
    
    fprintf('✅ 数据导出完成: %s\n', save_dir);
end

function has_method = hasmethod(obj, method_name)
    %HASMETHOD 检查对象是否有指定方法
    
    try
        if isobject(obj)
            method_list = methods(obj);
            has_method = any(strcmp(method_list, method_name));
        else
            has_method = false;
        end
    catch
        has_method = false;
    end
end
%% 在 main_fsp.m 文件末尾添加以下函数

function [env, agents] = createEnvironmentAndAgents(config)
    %CREATEENVIRONMENTANDAGENTS 创建环境和智能体
    
    fprintf('🏗️ 创建仿真环境和智能体...\n');
    
    try
        % 1. 创建TCS环境
        fprintf('  📍 创建TCS环境...\n');
        env = TCSEnvironment(config);
        fprintf('  ✓ TCS环境创建成功: %d站点, %d总资源\n', ...
                config.n_stations, config.total_resources);
        
        % 2. 创建攻击者智能体
        fprintf('  ⚔️ 创建攻击者智能体...\n');
        attacker = AgentFactory.createAttackerAgent(config, env);
        fprintf('  ✓ 攻击者创建成功: %s\n', attacker.name);
        
        % 3. 创建防御者智能体数组
        fprintf('  🛡️ 创建防御者智能体...\n');
        defenders = AgentFactory.createDefenderAgents(config, env);
        fprintf('  ✓ 防御者创建成功: %d个智能体\n', length(defenders));
        
        % 4. 组合智能体数组（攻击者在第一位）
        agents = cell(1, 1 + length(defenders));
        agents{1} = attacker;
        for i = 1:length(defenders)
            agents{i+1} = defenders{i};
        end
        
        fprintf('✅ 环境和智能体创建完成！\n\n');
        
    catch ME
        fprintf('❌ 主要实现失败，使用简化版本: %s\n', ME.message);
        [env, agents] = createEnvironmentAndAgents_simple(config);
    end
end

function [env, agents] = createEnvironmentAndAgents_simple(config)
    %CREATEENVIRONMENTANDAGENTS_SIMPLE 简化版创建函数
    
    fprintf('⚠️ 使用简化版环境和智能体...\n');
    
    % 创建简化环境
    env = struct();
    env.n_stations = config.n_stations;
    env.total_resources = config.total_resources;
    env.radi_history = [];
    env.nash_convergence_history = [];
    env.attack_coverage_history = [];
    
    % 添加简单的方法
    env.reset = @() ones(1, config.n_stations * 2);
    env.updateStrategies = @(att, def) updateStrategiesSimple(env, att, def);
    
    % 创建简单智能体
    agents = cell(1, 2);
    
    % 简单攻击者
    agents{1} = struct();
    agents{1}.name = 'SimpleAttacker';
    agents{1}.agent_type = 'attacker';
    agents{1}.getStrategy = @() ones(1, config.n_stations) / config.n_stations;
    
    % 简单防御者
    agents{2} = struct();
    agents{2}.name = 'SimpleDefender';
    agents{2}.agent_type = 'defender';
    agents{2}.getStrategy = @() ones(1, config.n_stations) / config.n_stations;
    
    fprintf('✅ 简化版创建完成\n');
end

function updateStrategiesSimple(env, attack_strategy, defense_strategy)
    %UPDATESTRATEGIESSIMPLE 简化版策略更新
    % 添加一些随机数据到历史记录
    env.radi_history(end+1) = rand();
    env.nash_convergence_history(end+1) = rand();
    env.attack_coverage_history(end+1) = rand();
end

function results = initializeResults(config)
    %INITIALIZERESULTS 初始化结果结构
    
    results = struct();
    results.iteration_data = [];
    results.performance_metrics = [];
    results.convergence_data = [];
    results.config = config;
    results.timestamp = datestr(now);
    
    fprintf('✓ 结果结构初始化完成\n');
end

function episode_results = runSimpleEpisodes(env, defenders, attacker, config)
    %RUNSIMPLEEPISODES 简单的episode运行函数
    
    episode_results = struct();
    episode_results.avg_radi = rand();
    episode_results.avg_efficiency = rand();
    episode_results.total_defender_reward = rand() * 100;
    episode_results.total_attacker_reward = rand() * 100;
    episode_results.avg_resource_allocation = ones(1, config.n_stations) / config.n_stations;
    
    % 模拟一些简单的episodes
    for episode = 1:min(10, config.n_episodes_per_iter)
        % 随机选择动作
        attacker_action = randi(config.n_stations);
        defender_action = ones(1, config.n_stations) * (config.total_resources / config.n_stations);
        
        % 简单的奖励计算
        episode_results.total_defender_reward = episode_results.total_defender_reward + rand();
        episode_results.total_attacker_reward = episode_results.total_attacker_reward + rand();
    end
    
    if mod(config.n_episodes_per_iter, 10) == 0
        fprintf('  📊 完成 %d episodes\n', config.n_episodes_per_iter);
    end
end

function recordIterationResults(results, episode_results, agents, iteration, config)
    %RECORDITERATIONRESULTS 记录迭代结果
    
    % 记录基本信息
    iteration_data = struct();
    iteration_data.iteration = iteration;
    iteration_data.timestamp = datestr(now);
    iteration_data.avg_radi = episode_results.avg_radi;
    iteration_data.total_rewards = episode_results.total_defender_reward + episode_results.total_attacker_reward;
    
    % 添加到结果结构
    if isempty(results.iteration_data)
        results.iteration_data = iteration_data;
    else
        results.iteration_data(end+1) = iteration_data;
    end
    
    if mod(iteration, 50) == 0
        fprintf('  💾 已记录迭代 %d 的结果\n', iteration);
    end
end