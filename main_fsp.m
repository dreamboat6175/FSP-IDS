%% main_fsp.m - FSP-TCS智能防御系统主仿真程序
% =========================================================================
% 描述: 简洁的主函数，调用其他模块完成仿真
% =========================================================================

function main_fsp()
    % FSP-TCS主程序入口
    
    clc; clear; close all;
    addpath(genpath(pwd));
    
    try
        %% === 1. 系统初始化 ===
        fprintf('\n=== FSP-TCS 智能防御系统仿真 ===\n');
        
        % 加载配置
        config = ConfigManager.loadConfig();
        
        % 初始化日志
        Logger.initialize(config.output.log_file, 'INFO');
        Logger.info('FSP-TCS仿真开始');
        
        %% === 2. 创建环境和智能体 ===
        env = TCSEnvironment(config);
        defender_agents = AgentFactory.createDefenderAgents(config, env);
        attacker_agent = AgentFactory.createAttackerAgent(config, env);
        monitor = PerformanceMonitor(config.n_iterations, length(defender_agents), config);
        
        %% === 3. 运行FSP仿真 ===
        Logger.info('开始FSP仿真训练...');
        results = FSPSimulator.run(env, defender_agents, attacker_agent, config, monitor);
        
        %% === 4. 生成可视化报告 ===
        % 收集所有智能体
        all_agents = {attacker_agent};
        for i = 1:length(defender_agents)
            all_agents{end+1} = defender_agents{i};
        end
        
        % 一行代码生成所有可视化内容
        generateVisualizationReport(all_agents, config);
        
        %% === 5. 保存结果 ===
        DataManager.saveResults(results, config, struct('defenders', defender_agents, 'attacker', attacker_agent));
        
        Logger.info('FSP-TCS仿真成功完成');
        Logger.close();
        
    catch ME
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
end