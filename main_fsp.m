%% main_fsp.m - 简洁的FSP-TCS智能防御系统主仿真程序
% =========================================================================
% 描述: 简洁的主函数，只负责调用其他函数，不做具体实现
% 每轮输出: 攻击者策略、三种防御者策略和性能、完整的可视化报告
% =========================================================================

function main_fsp()
    % FSP-TCS主程序入口 - 保持简洁，只负责调用
    
    clc; clear; close all;
    addpath(genpath(pwd));
    
    try
        %% === 1. 系统初始化 ===
        fprintf('\n=== FSP-TCS 智能防御系统仿真 ===\n');
        config = initializeSystem();
        
        %% === 2. 创建环境和智能体 ===
        [env, agents, monitor] = createEnvironmentAndAgents(config);
        
        %% === 3. 运行FSP仿真 ===
        results = runFSPSimulation(env, agents, monitor, config);
        
        %% === 4. 生成完整可视化报告 ===
        generateCompleteVisualizationReport(agents, results, config, env);
        
        %% === 5. 保存结果并清理 ===
        finalizeSimulation(results, config);
        
    catch ME
        handleSimulationError(ME);
    end
end

%% ========== 辅助函数实现 ==========

function config = initializeSystem()
    % 系统初始化
    config = ConfigManager.loadConfig();
    Logger.initialize(config.output.log_file, 'INFO');
    Logger.info('FSP-TCS仿真开始');
    fprintf('✓ 系统初始化完成\n');
end

function [env, agents, monitor] = createEnvironmentAndAgents(config)
    % 创建环境和智能体
    env = TCSEnvironment(config);
    defender_agents = AgentFactory.createDefenderAgents(config, env);
    attacker_agent = AgentFactory.createAttackerAgent(config, env);
    monitor = PerformanceMonitor(config.n_iterations, length(defender_agents), config);
    
    % 组织智能体结构
    agents = struct();
    agents.attacker = attacker_agent;
    agents.defenders = defender_agents;
    
    fprintf('✓ 环境和智能体创建完成\n');
    fprintf('  - 防御者数量: %d\n', length(defender_agents));
    fprintf('  - 攻击者: 1个\n');
end

function results = runFSPSimulation(env, agents, monitor, config)
    % 运行FSP仿真
    Logger.info('开始FSP仿真训练...');
    fprintf('🚀 开始FSP仿真训练...\n');
    
    results = FSPSimulator.run(env, agents.defenders, agents.attacker, config, monitor);
    
    fprintf('✓ FSP仿真训练完成\n');
    Logger.info('FSP仿真训练完成');
end

function generateCompleteVisualizationReport(agents, results, config, env)
    % 生成完整的可视化报告
    fprintf('\n📊 开始生成可视化报告...\n');
    
    % 收集所有智能体数据
    all_agents = {agents.attacker};
    for i = 1:length(agents.defenders)
        all_agents{end+1} = agents.defenders{i};
    end
    
    % 调用可视化报告生成器
    EnhancedVisualization.generateFullReport(all_agents, results, config, env);
    
    fprintf('✓ 可视化报告生成完成\n');
end

function finalizeSimulation(results, config)
    % 保存结果并清理
    DataManager.saveResults(results, config);
    Logger.info('FSP-TCS仿真成功完成');
    Logger.close();
    fprintf('✅ 仿真成功完成，结果已保存\n');
end

function handleSimulationError(ME)
    % 错误处理
    fprintf('❌ 仿真失败: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('错误位置: %s, 行号: %d\n', ME.stack(1).file, ME.stack(1).line);
    end
    
    if exist('Logger', 'class') && Logger.isInitialized()
        Logger.error(sprintf('仿真出错: %s', ME.message));
        if ~isempty(ME.stack)
            Logger.error(sprintf('错误位置: %s, 行号: %d', ME.stack(1).file, ME.stack(1).line));
        end
        Logger.close();
    end
    
    rethrow(ME);
end