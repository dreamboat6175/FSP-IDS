%% main_fsp.m - 修复版FSP-TCS主仿真程序
% =========================================================================
% 描述: 解决Logger静态方法调用问题的主仿真程序
% 版本: v2.1 - 修复Logger调用错误
% =========================================================================

function main_fsp()
    
    clc; clear; close all;
    
    % 添加所有子目录到路径
    addpath(genpath(pwd));

    try
        %% === 1. 系统初始化 ===
        fprintf('\n=== FSP-TCS 智能防御系统仿真 v2.1 ===\n');
        fprintf('正在初始化系统...\n\n');
        
        % 加载配置 - 检查ConfigManager是否存在
        if exist('ConfigManager', 'class')
            config = ConfigManager.loadConfig();
            ConfigManager.displayConfigSummary(config);
        else
            % 使用备用配置
            fprintf('ConfigManager不存在，使用备用配置\n');
            config = createBackupConfig();
        end
        
        % 初始化日志系统 - 使用修复后的Logger
        log_file = config.output.log_file;
        Logger.initialize(log_file, 'INFO');
        Logger.info('FSP-TCS仿真开始');
        
        % 设置随机种子
        if isfield(config, 'random_seed')
            rng(config.random_seed);
            Logger.info(sprintf('随机种子设置为: %d', config.random_seed));
        end
        
        %% === 2. 环境和智能体初始化 ===
        fprintf('正在创建环境和智能体...\n');
        
        % 创建环境
        if exist('TCSEnvironment', 'class')
            env = TCSEnvironment(config);
            Logger.info('TCS环境创建完成');
        else
            error('TCSEnvironment类不存在，请检查核心文件');
        end
        
        % 创建智能体
        if exist('AgentFactory', 'class')
            defender_agents = AgentFactory.createDefenderAgents(config, env);
            attacker_agent = AgentFactory.createAttackerAgent(config, env);
            Logger.info(sprintf('智能体创建完成: %d个防御者, 1个攻击者', length(defender_agents)));
        else
            error('AgentFactory类不存在，请检查工具文件');
        end
        
        % 创建性能监控器
        if exist('PerformanceMonitor', 'class')
            monitor = PerformanceMonitor(config.n_iterations, length(defender_agents), config);
            Logger.info('性能监控器初始化完成');
        else
            fprintf('PerformanceMonitor不存在，跳过性能监控\n');
            monitor = [];
        end
        
        %% === 3. FSP仿真主循环 ===
        fprintf('开始FSP仿真训练...\n');
        
        % 初始化结果存储
        results = initializeResults(config, length(defender_agents));
        
        % 仿真主循环
        for iteration = 1:config.n_iterations
            tic;
            
            % 运行一轮episodes
            if exist('FSPSimulator', 'class')
                episode_results = FSPSimulator.runIteration(env, defender_agents, attacker_agent, config, monitor);
            else
                % 使用简化的episode运行
                episode_results = runSimpleEpisodes(env, defender_agents, attacker_agent, config);
            end
            
            % 记录结果
            results = recordIterationResults(results, episode_results, iteration);
            
            % 更新性能监控
            if ~isempty(monitor)
                updatePerformanceMonitor(monitor, iteration, episode_results, config);
            end
            
            % 动态更新学习参数
            if mod(iteration, config.performance.param_update_interval) == 0
                if exist('ConfigManager', 'class')
                    config = ConfigManager.updateLearningParameters(config, iteration);
                else
                    config = updateLearningParametersSimple(config, iteration);
                end
                updateAgentParameters(defender_agents, attacker_agent, config);
            end
            
            % 显示进度
            iteration_time = toc;
            handleIterationOutput(iteration, config, iteration_time, episode_results);
            
            % 保存检查点
            if mod(iteration, config.output.checkpoint_interval) == 0 && config.output.save_checkpoints
                saveCheckpoint(defender_agents, attacker_agent, results, iteration, config);
            end
        end
        
        %% === 4. 结果分析和可视化 ===
        fprintf('\n仿真完成，正在生成分析报告...\n');
        
        % 保存最终结果
        if exist('DataManager', 'class')
            % 构造智能体结构
            agents_struct = struct();
            agents_struct.defenders = defender_agents;
            agents_struct.attacker = attacker_agent;
            
            DataManager.saveResults(results, config, agents_struct);
            Logger.info('仿真结果已保存');
        else
            % 使用简单保存方法
            saveResultsSimple(results, config);
        end
        % 生成可视化报告
        if config.output.visualization
            if exist('EnhancedVisualization', 'class')
                visualization = EnhancedVisualization(results, config, env);
                visualization.generateCompleteReport();
                Logger.info('可视化报告生成完成');
            else
                fprintf('EnhancedVisualization不存在，跳过可视化\n');
            end
        end
        
        % 生成文本报告
        if config.output.generate_report
            if exist('ReportGenerator', 'class')
                ReportGenerator.generateTextReport(results, config, monitor);
                Logger.info('文本报告生成完成');
            else
                generateSimpleReport(results, config);
            end
        end
        
        %% === 5. 系统清理 ===
        fprintf('\n=== 仿真完成 ===\n');
        printFinalSummary(results, config);
        
        Logger.info('FSP-TCS仿真成功完成');
        Logger.close();
        
    catch ME
        % 错误处理
        fprintf('\n❌ 仿真过程中发生错误:\n');
        fprintf('错误信息: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('错误位置: %s, 行号: %d\n', ME.stack(1).file, ME.stack(1).line);
        end
        
        % 记录错误日志
        if Logger.isInitialized()
            Logger.error(sprintf('仿真出错: %s', ME.message));
            if ~isempty(ME.stack)
                Logger.error(sprintf('错误位置: %s, 行号: %d', ME.stack(1).file, ME.stack(1).line));
            end
            Logger.close();
        end
        
        rethrow(ME);
    end
end
